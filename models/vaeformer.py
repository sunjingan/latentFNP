import warnings
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from typing import Optional

import modules

from modules.vit_nlc import Encoder, Decoder
from utils.distributions import DiagonalGaussianDistribution
from collections import OrderedDict

from torch.nn.utils import clip_grad_norm_
from functools import partial
from einops import rearrange
import utils.misc as utils
from utils.metrics import WRMSE
from utils.builder import get_optimizer, get_lr_scheduler

from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

import pandas as pd
class VAEformer(nn.Module):
    """
    Args:
    N (int): Number of channels
    M (int): Number of channels in the expansion layers (last layer of the
    encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, 
                 model_version,
                 embed_dim=None, 
                 z_channels=None,
                 y_channels=None,
                 sample_posterior=None, 
                 pretrained_vae=None, 
                 frozen_encoder=None, 
                 ddconfig=None, 
                 priorconfig=None,
                 rate_distortion_loss=None, 
                 kl_loss=None, 
                 ignore_keys:list=[], 
                 lower_dim= False,
                 **kwargs):

        if model_version == 69: #69 varibles
            embed_dim=256
            z_channels=256
            y_channels=1024
            lower_dim=True
            sample_posterior =False
            pretrained_vae = None #'./exp/comp/era5_autoencoder_ps10_159v/iter_150000.pth',
            frozen_encoder=False

            
        super().__init__(**kwargs)
        self.sample_posterior = sample_posterior
        self.lower_dim = lower_dim
        self.frozen_encoder = frozen_encoder
        ddconfig=dict(
                arch = 'vit_large',
                pretrained_model = '',
                patch_size=(11,10),
                patch_stride=(10,10),
                in_chans=69,
                out_chans=69,
                kwargs=dict( #will update the default vit config
                    z_dim =  256,
                    depth = 24,
                    embed_dim = 1024,
                    learnable_pos= True,
                    window= True,
                    window_size = [(24, 24), (12, 48), (48, 12)],
                    interval = 4,
                    drop_path_rate= 0.,
                    round_padding= True,
                    pad_attn_mask= True , # to_do: ablation
                    test_pos_mode= 'learnable_simple_interpolate', # to_do: ablation
                    lms_checkpoint_train= True,
                    img_size= (721, 1440)
                ))
        
        print(ddconfig)
        
        self.g_a = Encoder(**ddconfig)
        self.g_s = Decoder(**ddconfig)


    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""

        variable_num = state_dict["backbone.g_a.patch_embed.proj.weight"].size(1)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'kl_loss.logvar' not in k:
                new_state_dict[k.replace("backbone.", "")] = v
            

        net = cls(variable_num)
        # net.update(force=True)
        
        net.load_state_dict(new_state_dict)
        
        return net

    def init_from_ckpt(self, ckpt, ignore_keys=list()):
        last_saved: Optional[str]
        if isinstance(ckpt, str):
            if ckpt.endswith('.pth'):
                last_saved = ckpt
            else:
                save_file = osp.join(ckpt, 'last_checkpoint')

                if osp.exists(save_file):
                    with open(save_file) as f:
                        last_saved = f.read().strip()
                else:
                    raise ValueError(f"You do not have a saved checkpoint to restore, "
                                    f"please set the load path: {ckpt} as None in config file")

            sd = torch.load(last_saved, map_location="cpu")["state_dict"]
        else:
            sd=ckpt
            
        ga_state_dict = OrderedDict()
        gs_state_dict = OrderedDict()
        #quant_conv_state = OrderedDict()
        #post_quant_conv_state = OrderedDict()
        
        loss_state_dict = OrderedDict()

        for k, v in sd.items(): #if k in model_dict.keys()
            skip = [ True  for ik in ignore_keys if k.startswith(ik)]
            if len(skip) > 0:
                print("Deleting key {} from state_dict.".format(k))
                continue
            if 'encoder' in k:
                ga_state_dict[k.replace("backbone.encoder.", "")] = v
            if 'decoder' in k:
                gs_state_dict[k.replace("backbone.decoder.", "")] = v
            if 'logvar' in k:
                loss_state_dict[k.replace("backbone.loss.", "")] = v
            

        self.g_a.load_state_dict(ga_state_dict, strict=True)
        self.g_s.load_state_dict(gs_state_dict, strict=True)
        if self.frozen_encoder:
            for param in self.g_a.parameters():
                param.requires_grad = False


        print(f"Restored from {last_saved}, and make the frozen_encoder as {self.frozen_encoder}" )
    
    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def training_step(self, inputs, batch_idx, optimizer_idx):
        out_net = self(inputs)
        out_criterion = self.criterion(out_net, inputs)
        discloss = self.kl_loss(inputs, out_net['x_hat'], out_net['posterior'],
                                optimizer_idx, 0,
                                last_layer=self.get_last_layer(), split="train")


        return  {**discloss, **out_criterion, "aux_loss":self.aux_loss()}
    def prediction(self, inputs):

        t1 = time.time()
        out = self.compress(inputs)
        t2 =  time.time()
        x_hat = self.decompress(out['strings'], out['shape'])
        t3 =  time.time()


        return {
            **x_hat,
            "strings":out['strings'],
            "z_shape":out['shape'],
            'x_shape':inputs.shape,
            'encoding_time':(t2-t1)/inputs.size(0),
            'decoding_time':(t3-t2)/inputs.size(0)}
    
    def encode_latent(self, x):
        moments = self.g_a(x)
        posterior = None
        
        posterior = DiagonalGaussianDistribution(moments)
        if self.sample_posterior:
            y = posterior.sample()
        else:
            y = posterior.mode()
            
        return y
        
    def decode_latent(self, y):
        
        x_hat = self.g_s(y)
        
        return x_hat

    def forward(self, x):
        moments = self.g_a(x)
        posterior = None
        

        posterior = DiagonalGaussianDistribution(moments)
        if self.sample_posterior:
            y = posterior.sample()
        else:
            y = posterior.mode()
        
        
        #x_hat = self.g_s(y_hat)
        x_hat = self.g_s(y)

        return x_hat, moments
        
    
class VAE(object):
    def __init__(self, **model_params) -> None:
        super().__init__()

        params = model_params.get('params', {})
        criterion = model_params.get('criterion', 'CNPFLoss')
        self.optimizer_params = model_params.get('optimizer', {})
        self.scheduler_params = model_params.get('lr_scheduler', {})

        self.kernel = VAEformer(model_version=69)
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
        
        #args.max_epoch = 1000
        for epoch in range(args.max_epoch):
            begin_time = time.time()
            self.kernel.train()
            
            for step, batch_data in enumerate(train_data_loader):
                
                
                x = batch_data[0][0].to(self.device)
                
                self.optimizer.zero_grad()
                with autocast():
                    x_hat, moments = self.kernel(x)
                    
                    mean, logvar = torch.chunk(moments, 2, dim=1)
                    logvar = torch.clamp(logvar, -30.0, 20.0)
                    std = torch.exp(0.5 * logvar)
                    var = torch.exp(logvar)
                
                    if isinstance(self.criterion, torch.nn.Module):
                        self.criterion.train()
                    mseloss = 0.5*self.criterion_mse(x_hat, x) 
                    klloss = 0.5 * torch.mean(torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2, 3])
                    #print(mseloss.shape,klloss.shape)
                    loss = mseloss + klloss.mean()
                #loss.backward()
                if not loss.requires_grad:
                    raise RuntimeError("Loss does not require gradients. Check computation graph.")
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

                    x = batch_data[0][0].to(self.device)
                    #x = F.pad(x, (48, 48, 24, 23), "constant", 0).to(self.device)
                    x_hat, moments = self.kernel(x)
                    mean, logvar = torch.chunk(moments, 2, dim=1)
                    logvar = torch.clamp(logvar, -30.0, 20.0)
                    std = torch.exp(0.5 * logvar)
                    var = torch.exp(logvar)

                    if isinstance(self.criterion, torch.nn.Module):
                        self.criterion.eval()
                    #loss = self.criterion(y_pred, y_target).item()
                    mseloss = 0.5*self.criterion_mse(x_hat, x) 
                    klloss = 0.5 * torch.mean(torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2, 3])
                    #print(mseloss.shape,klloss.shape)
                    loss = mseloss + klloss.mean()
                
                    total_loss += loss.item()
                    
                    

                    if ((step + 1) % 100 == 0) | (step+1 == valid_step):
                        logger.info(f'Valid epoch:[{epoch+1}/{args.max_epoch}], step:[{step+1}/{valid_step}], loss:[{loss}]')
        
            if (total_loss/valid_step) < self.best_loss:
                if utils.get_world_size() > 1 and utils.get_rank() == 0:
                    torch.save(self.kernel.module.g_a.state_dict(), f'{args.rundir}/best_encoder_32.pth')
                    torch.save(self.kernel.module.g_s.state_dict(), f'{args.rundir}/best_decoder_32.pth')
                elif utils.get_world_size() == 1:
                    torch.save(self.kernel.g_a.state_dict(), f'{args.rundir}/best_encoder_32.pth')
                    torch.save(self.kernel.g_s.state_dict(), f'{args.rundir}/best_decoder_32.pth')
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

                x = batch_data[0][0].to(self.device)
                #x = F.pad(x, (48, 48, 24, 23), "constant", 0).to(self.device)
                x_hat, moments = self.kernel(x)
                mean, logvar = torch.chunk(moments, 2, dim=1)
                logvar = torch.clamp(logvar, -30.0, 20.0)
                std = torch.exp(0.5 * logvar)
                var = torch.exp(logvar)

                if isinstance(self.criterion, torch.nn.Module):
                    self.criterion.eval()
                #loss = self.criterion(y_pred, y_target).item()
                
                mseloss = 0.5*self.criterion_mse(x_hat, x) 
                klloss = 0.5 * torch.mean(torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2, 3])
                #print(mseloss.shape,klloss.shape)
                loss = mseloss + klloss.mean()
                
                total_loss += loss.item()
                #print(x_hat.shape)
                #print(x.shape)
                x_pred = x_hat #rearrange(x_hat, 'b h w c -> b c h w')
                x_target = x #rearrange(x, 'b h w c -> b c h w')
                mae = self.criterion_mae(x_pred, x_target).item()
                mse = self.criterion_mse(x_pred, x_target).item()
                rmse = WRMSE(x_pred, x_target, self.data_std)

                total_mae += mae
                total_mse += mse
                total_rmse += rmse
                if ((step + 1) % 20 == 0) | (step+1 == test_step):
                    print(x_pred.shape,x_target.shape,x.shape)
                    self.save_sample(step,x_pred,x_target)
                    
                    logger.info(f'Valid step:[{step+1}/{test_step}], loss:[{loss}], MAE:[{mae}], MSE:[{mse}], RMSE:[{rmse}]')
                    break

        logger.info(f'Average loss:[{total_loss/test_step}], MAE:[{total_mae/test_step}], MSE:[{total_mse/test_step}]')
        logger.info(f'Average RMSE:[{total_rmse/test_step}]')
    def save_sample(self,step,x_pred,x_target):
        B = x_pred.shape[0]
        for i in range(B):
            data = x_pred[i].cpu().numpy()
            print(data.shape)
            stacked_data = data.reshape(-1, data.shape[2])
            pd.DataFrame(stacked_data).to_csv('recons_step{}_{}.csv'.format(step,i),index=False, header=False)

            data = x_target[i].cpu().numpy()
            stacked_data = data.reshape(-1, data.shape[2])
            pd.DataFrame(stacked_data).to_csv('truth_step{}_{}.csv'.format(step,i), index=False, header=False)

    def plot_sample(self,step,gt,recons):
        lat = np.linspace(-90, 90, 721)  # 纬度从 -90° 到 90°
        lon = np.linspace(-180, 180, 1440)  # 经度从 -180° 到 180°
        lon, lat = np.meshgrid(lon, lat)

        # 创建地图
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())  # 使用 PlateCarree 投影（圆柱投影）

        # 绘制热力图
        heatmap = ax.pcolormesh(lon, lat, gt, cmap='hot', transform=ccrs.PlateCarree())

        # 添加国家分界线
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        # 添加颜色条
        cbar = plt.colorbar(heatmap, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Variable Value')

        # 设置标题
        plt.title('Global Heatmap with Latitude and Longitude')


        ax.set_global()  # 确保地图显示完整的全球范围
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')  # 添加网格线

        # 调整地图的边界
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  # 设置地图范围为全球

        # 保存图像
        plt.savefig('global_heatmap.png', dpi=300, bbox_inches='tight')

