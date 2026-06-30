from torch import nn
from monai.networks.nets import BasicUNet
from monai.networks.nets import UNETR
from networks.UXNet_3D.network_backbone import UXNET
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.SwinUNETR.SwinUNETR import SwinUNETR
from networks.mednext.MedNext import MedNeXt
from monai.networks.nets.swin_unetr import SwinUNETR
from networks.UNesT.unest import UNesT
from networks.SwinSMT.src.models.swin_smt import SwinSMT
from networks.nnWNet.nnWNet import WNet3D
from networks.SuperLightNet.superlightnet import NormalU_Net
from networks.VSmTrans.VSmTrans import VSmixTUnet
from networks.PHNet.phnet import PHNet
from networks.SegMamba.segmamba import SegMamba


from networks.CAFSANet.CAFSANet import CAFSANet


def get3dmodel(network, in_channel, out_classes):
    ## UNet
    if network == 'UNet':
        model = BasicUNet(in_channels=in_channel, out_channels=out_classes)
        
   
    ## nnFormer
    elif network == 'nnFormer':
        model = nnFormer(
            input_channels=in_channel, 
            num_classes=out_classes)      
        
        
    ## SwinUNETR 
    elif network == 'SwinUNETR':
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_channel,
            out_channels=out_classes,
            feature_size=48,
            use_checkpoint=False)


    elif network == 'nnWNet':
        model = WNet3D(
            in_channel=in_channel,
            num_classes=out_classes,
        )

    elif network == 'SuperLightNet':
        model = NormalU_Net(
            init_channels=in_channel,
            class_nums=out_classes,
            depths_unidirectional='small',
        )
    
    elif network == 'SFProNet':
        model = SFProNet(
            in_channels=in_channel,
            out_channels=out_classes,
        )

    

        
    return model
