import torch 
from torch import Tensor 
from typing import Type,Any,Callable,Union,List,Optional
import torch.nn as nn 

from attention_layer import Attention_layer

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

def conv3x3(in_planes:int,out_planes:int,stride:int=1,groups:int=1,dilation:int=1)->nn.Conv2d:
    """3x3 convolutions with padding"""
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=dilation,groups=groups,
                     bias=False,dilation=dilation)

def conv1x1(in_planes:int,out_planes:int,stride:int=1)->nn.Conv2d:
    """1x1 convolution"""

    return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)

class BasicBlock(nn.Module):
    expansion:int=1

    def __init__(
            self,
            inplanes:int,
            planes:int,
            strides:int=1,
            downsample:Optional[nn.Module]=None,
            groups:int=1,
            base_width:int=64,
            dilation:int=1,
            norm_layer:Optional[Callable[...,nn.Module]]=None,
            attention:bool=False,
            num_heads:int=8,
            kernel_size:int=7,
            image_size:int=224,
            inference:bool=False
    )->None:
        super(BasicBlock,self).__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        if groups!=1 or base_width!=64:
            raise ValueError('BasicBlock only suppourts groups=1 and base_width=64')
        if dilation>1:
            raise NotImplementedError("Dilation >1 not suppourted in BasicBlock")

        self.conv1=conv3x3(inplanes,width,strides)
        self.bn1=norm_layer(width)
        self.relu=nn.ReLU(inplace=True)
        if not attention:
            self.conv2=conv3x3(planes,planes)
        else:
            self.conv2=Attention_layer(in_channels=width,num_heads=num_heads,kernel_size=kernel_size,image_size=image_size,inference=inference)
        
        self.bn2=norm_layer(width)
        self.conv3=conv1x1(width,planes*self.expansion)
        self.bn3=norm_layer(planes*self.expansion)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.stride=strides 

        
