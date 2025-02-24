import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from functools import partial
from einops import rearrange
import utils.misc as utils
from utils.metrics import WRMSE
from utils.builder import get_optimizer, get_lr_scheduler
import modules
from utils.distributions import DiagonalGaussianDistribution
from models.vaeformer import VAEformer
from models.FNP import FNP_model

from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint


class latentFNP_model(nn.Module):
    def __init__(
        self,
        model_params
    ):
        super().__init__()

        VAEparams = model_params.get('VAEparams', {})
        self.vaeformer = VAEformer(model_version=69) #whether frozen encoder. If frozen, load pretrained model
        FNPparams = model_params.get('FNPparams', {})
        
        self.FNP = FNP_model(**FNPparams)

        #self.get_sigma = Mlp()

    
        self.sample_posterior = False
    
    def get_latent(self,bkg):
        moments = self.vaeformer.g_a(bkg)
        posterior = None
        
        posterior = DiagonalGaussianDistribution(moments)
        if self.sample_posterior:
            y = posterior.sample()
        else:
            y = posterior.mode()
            
        return y
    
    def decode_latent(self, y):
        
        x_hat = self.vaeformer.g_s(y)
        
        return x_hat

    def forward(self,obs_coor,obs_value,bkg,logger):
        bkg_latent = self.get_latent(bkg)
        bkg_latent = rearrange(bkg_latent, 'b c h w -> b h w c')
        #print(obs_coor.shape,obs_value.shape,bkg_latent.shape)
        ana_latent = self.FNP(obs_coor,obs_value, bkg_latent,logger)
        ana_latent = modules.channels_to_2nd_dim(ana_latent.squeeze(0))
        #print('ana_latent',ana_latent.shape)
        ana_mean = self.decode_latent(ana_latent)
        #print(bkg_latent.shape,ana_latent.shape,ana_mean.shape)

        return ana_mean


class latentFNP(object):
    def __init__(self, **model_params) -> None:
        super().__init__()

        print('model_params:',model_params)
        criterion = model_params.get('criterion', 'CNPFLoss') #UnifyMAE
        self.optimizer_params = model_params.get('optimizer', {})
        self.scheduler_params = model_params.get('lr_scheduler', {})

        self.kernel = latentFNP_model(model_params)

        encoder_path = '/mnt/petrelfs/sunjingan/HighResFNP/compress/VAEformer/models/best_encoder.pth'
        decoder_path = '/mnt/petrelfs/sunjingan/HighResFNP/compress/VAEformer/models/best_decoder.pth'
    
        self.kernel.vaeformer.g_a.load_state_dict(torch.load(encoder_path))
        self.kernel.vaeformer.g_s.load_state_dict(torch.load(decoder_path))
        
        frozen_encoder = True
        if frozen_encoder:
            for param in self.kernel.vaeformer.g_a.parameters():
                param.requires_grad = False
            for param in self.kernel.vaeformer.g_s.parameters():
                param.requires_grad = False
            
        
        self.best_loss = 9999999
        self.criterion = self.get_criterion(criterion)
        self.criterion_mae = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()
        self.scaler = GradScaler() 

        if utils.is_dist_avail_and_initialized():
            self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
            if self.device == torch.device('cpu'):
                raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def get_criterion(self, loss_type):
        if loss_type == 'CNPFLoss':
            return modules.CNPFLoss()
        elif loss_type == 'NLLLossLNPF':
            return modules.NLLLossLNPF()
        elif loss_type == 'ELBOLossLNPF':
            return modules.ELBOLossLNPF()
        elif loss_type == 'SUMOLossLNPF':
            return modules.SUMOLossLNPF()
        else:
            raise NotImplementedError('Invalid loss type.')
    
    
    def train(self, train_data_loader, valid_data_loader, logger, args):
        

        train_step = len(train_data_loader)
        valid_step = len(valid_data_loader)
        self.optimizer = get_optimizer(self.kernel, self.optimizer_params)
        self.scheduler = get_lr_scheduler(self.optimizer, self.scheduler_params, total_steps=train_step*args.max_epoch)

        for epoch in range(args.max_epoch):
            begin_time = time.time()
            self.kernel.train()
            
            for step, batch_data in enumerate(train_data_loader):
                
                if step == 0:
                    inp_data = torch.cat([batch_data[0][0], batch_data[0][1]], dim=1).numpy()
                truth = batch_data[0][-1].to(self.device, non_blocking=True)
                predict_data = args.forecast_model.run(None, {'input':inp_data})[0][:,:truth.shape[1]]
                #predict_data is the bkg field. shape is 721*1440
                predict_data = torch.from_numpy(predict_data).to(self.device)

                #x_context y_context is the simulated observations. One can adjust the observation resolution
                truth_down = F.interpolate(truth, size=(128,256), mode='bilinear')
                x_context = rearrange(torch.rand(truth_down.shape, device=self.device) >= args.ratio, 'b c h w -> b h w c')
                y_context = rearrange(truth_down, 'b c h w -> b h w c')
                
                #the analysis truth
                y_target = truth #rearrange(truth, 'b c h w -> b h w c')
                
                self.optimizer.zero_grad()
                with autocast():
                    y_pred = self.kernel(x_context,y_context,predict_data,logger)

                    analysis = y_pred #modules.channels_to_2nd_dim(y_pred)
                    #reformulate the inp data
                    #print("ana inp:",inp_data.shape,analysis.shape)
                    inp_data = np.concatenate([inp_data[:,truth.shape[1]:], analysis.detach().cpu().numpy()], axis=1)
                
                    if isinstance(self.criterion, torch.nn.Module):
                        self.criterion.train()
                    loss = self.criterion_mse(analysis, y_target)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.kernel.parameters(), max_norm=1)
                #self.optimizer.step()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                
                
                
                if ((step + 1) % 100 == 0) | (step+1 == train_step):
                    logger.info(f'Train epoch:[{epoch+1}/{args.max_epoch}], step:[{step+1}/{train_step}], lr:[{self.scheduler.get_last_lr()[0]}], loss:[{loss.item()}]')

            self.kernel.eval()
            with torch.no_grad():
                total_loss = 0

                for step, batch_data in enumerate(valid_data_loader):

                    if step == 0:
                        inp_data = torch.cat([batch_data[0][0], batch_data[0][1]], dim=1).numpy()
                    truth = batch_data[0][-1].to(self.device, non_blocking=True)
                    #truth shape: 721*1440
                    predict_data = args.forecast_model.run(None, {'input':inp_data})[0][:,:truth.shape[1]]
                    predict_data = torch.from_numpy(predict_data).to(self.device)
                    
                    truth_down = F.interpolate(truth, size=(128,256), mode='bilinear')
                    x_context = rearrange(torch.rand(truth_down.shape, device=self.device) >= args.ratio, 'b c h w -> b h w c')
                    y_context = rearrange(truth_down, 'b c h w -> b h w c')

                    y_target = truth #rearrange(truth, 'b c h w -> b h w c')
                    #x_context y_context is the simulated observations.
                    
                    y_pred = self.kernel(x_context,y_context,predict_data,logger)
                    
                    analysis = y_pred
                    inp_data = np.concatenate([inp_data[:,truth.shape[1]:], analysis.detach().cpu().numpy()], axis=1)
                    

                    if isinstance(self.criterion, torch.nn.Module):
                        self.criterion.eval()
                    loss = self.criterion_mse(y_pred, y_target).item()
                    total_loss += loss
                    

                    if ((step + 1) % 100 == 0) | (step+1 == valid_step):
                        logger.info(f'Valid epoch:[{epoch+1}/{args.max_epoch}], step:[{step+1}/{valid_step}], loss:[{loss}]')
        
            if (total_loss/valid_step) < self.best_loss:
                if utils.get_world_size() > 1 and utils.get_rank() == 0:
                    torch.save(self.kernel.module.state_dict(), f'{args.rundir}/best_model.pth')
                elif utils.get_world_size() == 1:
                    torch.save(self.kernel.state_dict(), f'{args.rundir}/best_model.pth')
                logger.info(f'New best model appears in epoch {epoch+1}.')
                self.best_loss = total_loss/valid_step
            logger.info(f'Epoch {epoch+1} average loss:[{total_loss/valid_step}], time:[{time.time()-begin_time}]')

    def test(self, test_data_loader, logger, args):
        
        test_step = len(test_data_loader)
        data_mean, data_std = test_data_loader.dataset.get_meanstd()
        self.data_std = data_std.to(self.device)

        self.kernel.eval()
        with torch.no_grad():
            total_loss = 0
            total_mae = 0
            total_mse = 0
            total_rmse = 0

            for step, batch_data in enumerate(test_data_loader):

                input_list, y_target = self.process_data(batch_data[0], args)
                y_pred = self.kernel(input_list)
                if isinstance(self.criterion, torch.nn.Module):
                    self.criterion.eval()
                loss = self.criterion(y_pred, y_target).item()
                
                y_pred = rearrange(y_pred[0].mean[0], 'b h w c -> b c h w')
                y_target = rearrange(y_target, 'b h w c -> b c h w')
                mae = self.criterion_mae(y_pred, y_target).item()
                mse = self.criterion_mse(y_pred, y_target).item()
                rmse = WRMSE(y_pred, y_target, self.data_std)

                total_loss += loss
                total_mae += mae
                total_mse += mse
                total_rmse += rmse
                if ((step + 1) % 100 == 0) | (step+1 == test_step):
                    logger.info(f'Valid step:[{step+1}/{test_step}], loss:[{loss}], MAE:[{mae}], MSE:[{mse}]')

        logger.info(f'Average loss:[{total_loss/test_step}], MAE:[{total_mae/test_step}], MSE:[{total_mse/test_step}]')
        logger.info(f'Average RMSE:[{total_rmse/test_step}]')