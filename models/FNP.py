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


class obsEncoder(nn.Module):

    def __init__(
        self,
        n_channels=[4,13,13,13,13,13],
        r_dim=64,
        XEncoder=nn.Identity,
        Conv=lambda y_dim: modules.make_abs_conv(nn.Conv2d)(
            y_dim,
            y_dim,
            groups=y_dim,
            kernel_size=11,
            padding=11 // 2,
            bias=False,
        ),
        CNN=partial(
            modules.CNN,
            ConvBlock=modules.ResConvBlock,
            Conv=nn.Conv2d,
            n_blocks=12,
            Normalization=nn.BatchNorm2d,
            activation=nn.SiLU(),
            is_chan_last=True,
            kernel_size=9,
            n_conv_layers=2,
        )):
        super().__init__()

        self.r_dim = r_dim
        
        self.n_channels = n_channels
        self.x_encoder = XEncoder()

        # components for encode_globally
        self.conv = [Conv(y_dim) for y_dim in n_channels]  # for each single channel
        self.conv.append(modules.make_abs_conv(nn.Conv2d)(
            in_channels=sum(n_channels),
            out_channels=sum(n_channels),
            groups=1,
            kernel_size=11,
            padding=11 // 2,
            bias=False,
        ))  # for all channels
        self.conv = nn.ModuleList(self.conv)

        self.resizer = [nn.Linear(y_dim * 2, self.r_dim) for y_dim in n_channels]  # 2 because also confidence channels
        self.resizer.append(nn.Linear(sum(n_channels) * 2, self.r_dim))
        self.resizer = nn.ModuleList(self.resizer)

        self.induced_to_induced = nn.ModuleList([CNN(self.r_dim) for _ in range(len(n_channels)+1)])

    def forward(self, X_cntxt, Y_cntxt):

        X_cntxt = self.x_encoder(X_cntxt)  # b,h,w,c
        
        # {R^u}_u
        # size = [batch_size, *n_rep, r_dim] for n_channels list
        R_trgt = self.encode_globally(X_cntxt, Y_cntxt)

        return R_trgt

    def cntxt_to_induced(self, mask_cntxt, X, index):
        """Infer the missing values  and compute a density channel."""

        # channels have to be in second dimension for convolution
        # size = [batch_size, y_dim, *grid_shape]
        X = modules.channels_to_2nd_dim(X)
        # size = [batch_size, x_dim, *grid_shape]
        mask_cntxt = modules.channels_to_2nd_dim(mask_cntxt).float()

        # size = [batch_size, y_dim, *grid_shape]
        X_cntxt = X * mask_cntxt
        signal = self.conv[index](X_cntxt)
        density = self.conv[index](mask_cntxt.expand_as(X))

        # normalize
        out = signal / torch.clamp(density, min=1e-5)

        # size = [batch_size, y_dim * 2, *grid_shape]
        out = torch.cat([out, density], dim=1)

        # size = [batch_size, *grid_shape, y_dim * 2]
        out = modules.channels_to_last_dim(out)

        # size = [batch_size, *grid_shape, r_dim]
        out = self.resizer[index](out)

        return out

    def encode_globally(self, mask_cntxt, X):

        # size = [batch_size, *grid_shape, r_dim] for each single channel
        R_induced_all = []
        for i in range(len(self.n_channels)):
            R_induced = self.cntxt_to_induced(mask_cntxt[...,sum(self.n_channels[:i]):sum(self.n_channels[:i+1])], 
                                              X[...,sum(self.n_channels[:i]):sum(self.n_channels[:i+1])], i)
            R_induced = self.induced_to_induced[i](R_induced)
            R_induced_all.append(R_induced)
        # the last for all channels
        R_induced = self.cntxt_to_induced(mask_cntxt, X, len(self.n_channels))
        R_induced = self.induced_to_induced[len(self.n_channels)](R_induced)
        R_induced_all.append(R_induced)

        return R_induced_all


class bkgEncoder(nn.Module):

    def __init__(
        self,
        n_channels=[16,48,48,48,48,48],
        r_dim=64,
        XEncoder=nn.Identity,
        Conv=lambda y_dim: modules.make_abs_conv(nn.Conv2d)(
            y_dim,
            y_dim,
            groups=y_dim,
            kernel_size=11,
            padding=11 // 2,
            bias=False,
        ),
        CNN=partial(
            modules.CNN,
            ConvBlock=modules.ResConvBlock,
            Conv=nn.Conv2d,
            n_blocks=12,
            Normalization=nn.BatchNorm2d,
            activation=nn.SiLU(),
            is_chan_last=True,
            kernel_size=9,
            n_conv_layers=2,
        )):
        super().__init__()

        self.r_dim = r_dim
        
        self.n_channels = n_channels
        self.x_encoder = XEncoder()

        # components for encode_globally
        self.conv = [Conv(y_dim) for y_dim in n_channels]  # for each single channel
        self.conv.append(modules.make_abs_conv(nn.Conv2d)(
            in_channels=sum(n_channels),
            out_channels=sum(n_channels),
            groups=1,
            kernel_size=11,
            padding=11 // 2,
            bias=False,
        ))  # for all channels
        self.conv = nn.ModuleList(self.conv)

        self.resizer = [nn.Linear(y_dim * 2, self.r_dim) for y_dim in n_channels]  # 2 because also confidence channels
        self.resizer.append(nn.Linear(sum(n_channels) * 2, self.r_dim))
        self.resizer = nn.ModuleList(self.resizer)

        self.induced_to_induced = nn.ModuleList([CNN(self.r_dim) for _ in range(len(n_channels)+1)])

    def forward(self, Y_cntxt):

        # {R^u}_u
        # size = [batch_size, *n_rep, r_dim] for n_channels list
        R_trgt = self.encode_globally(Y_cntxt)

        return R_trgt

    def cntxt_to_induced(self, X, index):
        """Infer the missing values  and compute a density channel."""

        # channels have to be in second dimension for convolution
        # size = [batch_size, y_dim, *grid_shape]
        X = modules.channels_to_2nd_dim(X)
        
        # size = [batch_size, y_dim, *grid_shape]
        X_cntxt = X
        signal = self.conv[index](X_cntxt)
        #mask_cntxt all true
        density = self.conv[index](mask_cntxt.expand_as(X))

        # normalize
        out = signal / torch.clamp(density, min=1e-5)

        # size = [batch_size, y_dim * 2, *grid_shape]
        out = torch.cat([out, density], dim=1)

        # size = [batch_size, *grid_shape, y_dim * 2]
        out = modules.channels_to_last_dim(out)

        # size = [batch_size, *grid_shape, r_dim]
        out = self.resizer[index](out)

        return out

    def encode_globally(self, X):

        # size = [batch_size, *grid_shape, r_dim] for each single channel
        R_induced_all = []
        for i in range(len(self.n_channels)):
            R_induced = self.cntxt_to_induced(X[...,sum(self.n_channels[:i]):sum(self.n_channels[:i+1])], i)
            R_induced = self.induced_to_induced[i](R_induced)
            R_induced_all.append(R_induced)
        # the last for all channels
        R_induced = self.cntxt_to_induced(X, len(self.n_channels))
        R_induced = self.induced_to_induced[len(self.n_channels)](R_induced)
        R_induced_all.append(R_induced)

        return R_induced_all

class FNP_model(nn.Module):

    def __init__(
        self,
        n_channels=[4,13,13,13,13,13],
        r_dim=128,
        use_nfl=True,
        use_dam=True,
    ):
        super().__init__()

        self.r_dim = r_dim
        
        self.y_dim = sum(n_channels)
        self.n_channels = n_channels
        self.use_dam = use_dam

        if use_nfl:
            EnCNN = partial(
                modules.FCNN,
                ConvBlock=modules.ResConvBlock,
                Conv=nn.Conv2d,
                n_blocks=4,
                Normalization=nn.BatchNorm2d,
                activation=nn.SiLU(),
                is_chan_last=True,
                kernel_size=9,
                n_conv_layers=2)
        else:
            EnCNN = partial(
                modules.CNN,
                ConvBlock=modules.ResConvBlock,
                Conv=nn.Conv2d,
                n_blocks=12,
                Normalization=nn.BatchNorm2d,
                activation=nn.SiLU(),
                is_chan_last=True,
                kernel_size=9,
                n_conv_layers=2)
            
        Decoder=modules.discard_ith_arg(partial(modules.MLP, n_hidden_layers=4, hidden_size=self.r_dim), i=0)
        self.obs_encoder = obsEncoder(n_channels=[4,13,13,13,13,13], r_dim=self.r_dim, CNN=EnCNN)
        #after VAEformer, the bkg field is [b,256,72,144]
        #n_channels = [16,48,48,48,48,48]
        self.back_encoder = bkgEncoder(n_channels=[16,48,48,48,48,48], r_dim=self.r_dim, CNN=EnCNN)
        self.fusion = nn.ModuleList([nn.Linear(self.r_dim * 2, self.r_dim) for _ in range(len(n_channels)+1)])

        # times 2 out because loc and scale (mean and var for gaussian)
        # hear we only consider mean Jing-An
        n_channels = [16,48,48,48,48,48]
        self.decoder = nn.ModuleList([Decoder(y_dim, self.r_dim * 2, y_dim ) for y_dim in n_channels])
        if self.use_dam:
            self.smooth = nn.ModuleList([nn.Conv2d(self.r_dim * 2, self.r_dim, 9, padding=4) for _ in range(len(n_channels)+1)])

        self.reset_parameters()

    def reset_parameters(self):
        modules.weights_init(self)

    def forward(self, obs_coor,obs_value, bkg_latent,logger):
        Xo_cotxt, Yo_cntxt, Yb_cntxt = obs_coor,obs_value, bkg_latent
        #Xo_cntxt, Yo_cntxt, Xb_cntxt, Yb_cntxt, X_trgt = input_list[0], input_list[1], input_list[2], input_list[3], input_list[4]
        
        # {R^u}_u
        # size = [batch_size, *n_rep, r_dim] for n_channels list
        #logger.info("time before encoder")
        Ro_trgt = self.obs_encoder(Xo_cntxt, Yo_cntxt)
        Rb_trgt = self.back_encoder(Yb_cntxt)
        #logger.info("time after encoder")

        z_samples, q_zCc, q_zCct = None, None, None
    
        # interpolate
        Ro_trgt = [rearrange(Ro_trgt[i], 'b h w c -> b c h w') for i in range(len(self.n_channels)+1)]
        Rb_trgt = [rearrange(Rb_trgt[i], 'b h w c -> b c h w') for i in range(len(self.n_channels)+1)]
        Ro_trgt = [F.interpolate(Ro_trgt[i], size=Rb_trgt[i].shape[2:], mode='bilinear') for i in range(len(self.n_channels)+1)]
        Ro_trgt = [rearrange(Ro_trgt[i], 'b c h w -> b h w c') for i in range(len(self.n_channels)+1)]
        Rb_trgt = [rearrange(Rb_trgt[i], 'b c h w -> b h w c') for i in range(len(self.n_channels)+1)]
        
        # representation fusion
        R_fusion = [self.fusion[i](torch.cat([Ro_trgt[i], Rb_trgt[i]], dim=-1)) for i in range(len(self.n_channels)+1)]
        if self.use_dam:
            R_similar = [rearrange(self.similarity(R_fusion[i], Rb_trgt[i], Ro_trgt[i]), 'b h w c -> b c h w') for i in range(len(self.n_channels)+1)]
            R_fusion = [rearrange(self.smooth[i](R_similar[i]), 'b c h w -> b h w c') for i in range(len(self.n_channels)+1)]

        # size = [n_z_samples, batch_size, *n_trgt, r_dim]
        R_trgt = [self.trgt_dependent_representation(Xo_cntxt, Xb_cntxt, z_samples, R_fusion[i]) for i in range(len(self.n_channels)+1)]
    
        # p(y|cntxt,trgt)
        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        #logger.info("time before decoder")
        p_yCc = self.decode(X_trgt, R_trgt, Yb_cntxt)
        #logger.info("time after decoder")

        return p_yCc
    
    def similarity(self, R, Rb, Ro):

        distb = torch.sqrt(torch.sum((R-Rb)**2, dim=-1, keepdim=True))
        disto = torch.sqrt(torch.sum((R-Ro)**2, dim=-1, keepdim=True))
        mask = (disto > distb).float()
        R = torch.cat([Ro * mask + Rb * (1-mask), R], dim=-1)

        return R
    
    def trgt_dependent_representation(self, _, __, ___, R_induced):

        # n_z_samples=1. size = [1, batch_size, n_trgt, r_dim]
        return R_induced.unsqueeze(0)
    
    def decode(self,R_trgt, Yb_cntxt):

        locs = []
        
        for i in range(len(self.n_channels)):
            R_trgt_single = torch.cat([R_trgt[i], R_trgt[-1]], dim=-1)

            # size = [n_z_samples, batch_size, *n_trgt, y_dim]
            p_y_loc = self.decoder[i](_, R_trgt_single)
            print("p_y_loc:",p_y_loc.shape)

            locs.append(p_y_loc)

        locs = torch.cat(locs, dim=-1) + Yb_cntxt
        
        print("locs.shape:",locs.shape)
        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        

        return locs #has the same shape with the bkg field
