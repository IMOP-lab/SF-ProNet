import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
import pytorch_lightning as pl
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from typing import Optional, Sequence, Union
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
import pywt
from .KAN.KANLayer import KANLayer



class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return self._channels_last_norm(x)
        elif self.data_format == "channels_first":
            return self._channels_first_norm(x)
        else:
            raise NotImplementedError("Unsupported data_format: {}".format(self.data_format))

    def _channels_last_norm(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def _channels_first_norm(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class TwoConv(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
    ):
        super().__init__()

        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
    ):
        super().__init__()
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
    ):
        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x


class filter_trans(nn.Module):
    def __init__(self, mode='low'):
        super(filter_trans, self).__init__()
        self.mode = mode
    
    def forward(self, x):
        # Apply the wavelet transform
        coeffs = pywt.wavedecn(x.detach().cpu().numpy(), wavelet='db2', level=2, axes=(2, 3, 4))
        # Extract the low-frequency components
        low_freq_coeffs = coeffs[0]
        
        # Convert the low-frequency components back to a tensor
        low_freq_tensor = torch.tensor(low_freq_coeffs, dtype=x.dtype, device=x.device)
        return low_freq_tensor


class iSpaGate(nn.Module):
    def __init__(self, rate, feat):
        super(iSpaGate, self).__init__()
        self.rate = nn.Parameter(torch.tensor(rate), requires_grad=True)
        self.feat = feat

        self.low_freq_input_conv = nn.Conv3d(1, self.feat, kernel_size=1)

        self.low_freq_transform = nn.Sequential(
            nn.Conv3d(self.feat, self.feat, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(self.feat, self.feat, kernel_size=3, padding=1)
        )

        self.mask_generator = nn.Sequential(
            nn.Conv3d(self.feat, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, low_freq_input):

       low_freq_input = self.low_freq_input_conv(low_freq_input)
       low_freq_processed = self.low_freq_transform(low_freq_input)
       low_freq_processed_upsampled = F.interpolate(low_freq_processed, size=x.shape[2:], mode='trilinear', align_corners=False)
       mask = self.mask_generator(low_freq_processed_upsampled)
       mask = mask.repeat(1, x.shape[1], 1, 1, 1)
       x_masked = x * mask
       y = x_masked * self.rate + low_freq_processed_upsampled * (1 - self.rate)

       return y


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding3D, self).__init__()
        self.channels = channels

    def forward(self, x):
        b, c, d, h, w = x.shape
        device = x.device

        div_term = torch.exp(torch.arange(0, c, 2, device=device) * (-torch.log(torch.tensor(10000.0)) / c))
        
        dz = torch.arange(0, d, device=device).unsqueeze(1)
        pe_z = torch.zeros(d, c, device=device)
        pe_z[:, 0::2] = torch.sin(dz * div_term)
        pe_z[:, 1::2] = torch.cos(dz * div_term)

        dy = torch.arange(0, h, device=device).unsqueeze(1)
        pe_y = torch.zeros(h, c, device=device)
        pe_y[:, 0::2] = torch.sin(dy * div_term)
        pe_y[:, 1::2] = torch.cos(dy * div_term)

        dx = torch.arange(0, w, device=device).unsqueeze(1)
        pe_x = torch.zeros(w, c, device=device)
        pe_x[:, 0::2] = torch.sin(dx * div_term)
        pe_x[:, 1::2] = torch.cos(dx * div_term)

        pe_z = pe_z.unsqueeze(1).unsqueeze(1).repeat(1, h, w, 1)
        pe_y = pe_y.unsqueeze(0).unsqueeze(2).repeat(d, 1, w, 1)
        pe_x = pe_x.unsqueeze(0).unsqueeze(1).repeat(d, h, 1, 1)

        pe = pe_z + pe_y + pe_x
        pe = pe.permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, d, h, w]

        return x + pe

class FluFormer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1, device='cuda'):
        super(FluFormer, self).__init__()
        
        self.device = device

        self.positional_encoding = PositionalEncoding3D(hidden_size).to(self.device)

        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout).to(self.device)

        self.conv_branch = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(),
            nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(),
        ).to(self.device)

        self.kan1 = KANLayer(
            in_dim=hidden_size,
            out_dim=dim_feedforward,
            num=5,
            k=3,
            base_fun=nn.ReLU(),
            grid_eps=0.02,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            device=self.device,
        ).to(self.device)

        self.kan2 = KANLayer(
            in_dim=dim_feedforward,
            out_dim=hidden_size,
            num=5,
            k=3,
            base_fun=nn.ReLU(),
            grid_eps=0.02,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            device=self.device,
        ).to(self.device)

        self.norm1 = nn.InstanceNorm3d(hidden_size).to(self.device)
        self.norm2 = nn.LayerNorm(hidden_size).to(self.device)       
        self.dropout = nn.Dropout(dropout).to(self.device)

        self.fusion = nn.Sequential(
            nn.Conv3d(2 * hidden_size, hidden_size, kernel_size=1),
            nn.BatchNorm3d(hidden_size),
            nn.ReLU()
        ).to(self.device)
    
    def forward(self, x):

        x = x.to(self.device)
        x = self.positional_encoding(x)
        residual = x

        # Reshape x to [seq_len, batch_size, hidden_size] for multihead attention
        b, c, d, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(2, 0, 1)  # x_flat: [seq_len, batch_size, hidden_size]

        # Multihead Attention
        attn_output, _ = self.attention(x_flat, x_flat, x_flat)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, d, h, w)  # [b, c, d, h, w]

        conv_output = self.conv_branch(x)

        combined = torch.cat([attn_output, conv_output], dim=1)  
        x = self.fusion(combined)

        x = residual + self.dropout(x)
        x = self.norm1(x)

        # Reshape x for KANLayer
        x_flat = x.view(b, c, -1).permute(2, 0, 1)  # [seq_len, batch_size, hidden_size]
        seq_len, batch_size, hidden_size = x_flat.shape
        x_reshaped = x_flat.reshape(-1, hidden_size)  # [seq_len * batch_size, hidden_size]

        ff_output, _, _, _ = self.kan1(x_reshaped)
        ff_output, _, _, _ = self.kan2(ff_output)

        # Reshape back to [seq_len, batch_size, hidden_size]
        ff_output = ff_output.view(seq_len, batch_size, hidden_size)
        x_flat = x_flat + self.dropout(ff_output)
        x_flat = self.norm2(x_flat)

        # Reshape back to original 3D format
        x = x_flat.permute(1, 2, 0).view(b, c, d, h, w)

        return x


class eLAREGraph(nn.Module):
    def __init__(self, channels, hidden_dim=None, activation='relu', normalization='batch', num_layers=3):
        super(eLAREGraph, self).__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim if hidden_dim is not None else channels
        self.num_layers = num_layers  

        self.W1_list = [nn.Parameter(torch.Tensor(channels, self.hidden_dim)) for _ in range(num_layers)]
        self.W2_list = [nn.Parameter(torch.Tensor(self.hidden_dim, channels)) for _ in range(num_layers)]

        for i, (W1, W2) in enumerate(zip(self.W1_list, self.W2_list)):
            self.register_parameter(f'W1_{i}', W1)
            self.register_parameter(f'W2_{i}', W2)
            nn.init.xavier_uniform_(W1)
            nn.init.xavier_uniform_(W2)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise NotImplementedError(f"Activation '{activation}' is not implemented.")

        if normalization == 'batch':
            self.norm_list = nn.ModuleList([nn.BatchNorm3d(channels) for _ in range(num_layers)])
        elif normalization == 'layer':
            self.norm_list = nn.ModuleList([nn.LayerNorm([channels, 1, 1, 1]) for _ in range(num_layers)])
        else:
            self.norm_list = [None] * num_layers

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: [batch_size, channels, D, H, W]
        original_input = x  

        residual = x  

        batch_size, channels, D, H, W = x.shape
        N = D * H * W

        x_flat = x.view(batch_size, channels, N)  # [batch_size, channels, N]

        for i in range(self.num_layers):
            x_mean = x_flat.mean(dim=2)  # [batch_size, channels]

            x_norm = F.normalize(x_mean, p=2, dim=1)  # [batch_size, channels]
            similarity = torch.bmm(x_norm.unsqueeze(2), x_norm.unsqueeze(1))  # [batch_size, channels, channels]

            A = F.softmax(similarity, dim=-1)  # [batch_size, channels, channels]

            x_flat = torch.bmm(A, x_flat)         # [batch_size, channels, N]
            x_flat = torch.matmul(self.W1_list[i].T, x_flat)  # [batch_size, hidden_dim, N]
            x_flat = self.activation(x_flat)     
            x_flat = torch.matmul(self.W2_list[i].T, x_flat)  # [batch_size, channels, N]
            x_flat = self.dropout(x_flat)         # Dropout
            
            x = x_flat.view(batch_size, channels, D, H, W)

            if self.norm_list[i] is not None:
                x = self.norm_list[i](x)

            x = x + residual
            residual = x  

            x_flat = x.view(batch_size, channels, N)

        x = x + original_input

        return x



class SFProNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 64, 128, 256, 512, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.05,
        upsample: str = "deconv",
        depths=[2, 2, 2, 2],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 512,
        conv_block: bool = True,
        res_block: bool = True,
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)
        
        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.FluFormer_1 = FluFormer(hidden_size=fea[4], num_heads=8, dim_feedforward=2048, dropout=dropout)
        self.FluFormer_2 = FluFormer(hidden_size=fea[4], num_heads=8, dim_feedforward=2048, dropout=dropout)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.eLAREGraph_4 = eLAREGraph(channels=fea[3])
        self.eLAREGraph_3 = eLAREGraph(channels=fea[2])
        self.eLAREGraph_2 = eLAREGraph(channels=fea[1])
        self.eLAREGraph_1 = eLAREGraph(channels=fea[5])


        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

        self.iSpaGate1 = iSpaGate(0.5, 32)
        self.iSpaGate2 = iSpaGate(0.5, 64)
        self.iSpaGate3 = iSpaGate(0.5, 128)
        self.iSpaGate4 = iSpaGate(0.5, 256)
        self.iSpaGate5 = iSpaGate(0.5, 512)
        
        self.filter_trans = filter_trans('low')


    def forward(self, x: torch.Tensor):
 
        filter_low = self.filter_trans(x)

        x0 = self.conv_0(x)
        x0 = self.iSpaGate1(x0, filter_low) * x0
        x1 = self.down_1(x0)
        x1 = self.iSpaGate2(x1, filter_low) * x1
        x2 = self.down_2(x1)
        x2 = self.iSpaGate3(x2, filter_low) * x2
        x3 = self.down_3(x2)
        x3 = self.iSpaGate4(x3, filter_low) * x3
        x4 = self.down_4(x3)
        x4 = self.iSpaGate5(x4, filter_low) * x4
        
        x4 = self.FluFormer_1(x4)
        x4 = self.FluFormer_2(x4)

        u4 = self.upcat_4(x4, x3)
        u4 = self.eLAREGraph_4(u4)
        u3 = self.upcat_3(u4, x2)
        u3 = self.eLAREGraph_3(u3)
        u2 = self.upcat_2(u3, x1)
        u2 = self.eLAREGraph_2(u2)
        u1 = self.upcat_1(u2, x0)
        u1 = self.eLAREGraph_1(u1)

        logits = self.final_conv(u1)
        
        return logits
