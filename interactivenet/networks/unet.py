import warnings
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import numpy as np

class UNet(nn.Module):
    """ UNet.

        Parameters
        ----------
        spatial_dims: int
            number of dims in the input tensor. Currently only 3 possible (not yet 2D, 2.5D network inplemented)
        in_channels: int
            number of channels in the input tensor.
        out_channels: int
            the number of features in the output segmentation or
            size of feature space for barlow twins.
        kernel_size: list of tuples or int
            kernel_sizes for each layer. Also determines number oflayer
        strides: list of tuples or int
            strides for each layer.
        upsample_kernel_size: list of tuples or int
            upsample_kernel_size for each layer stride[1:].
        activation: 'str', default 'LRELU'
            can also provide PRELU and RELU instead.
        normalisation: 'str', default 'instance'
            can also provide batch normalisation instead.
        deep_supervision: bool, default True
            if you wish to apply deep supervision. At this time will happen at all feature spaces.
    """
    def __init__(
        self, 
        spatial_dims:int, 
        in_channels:int, 
        out_channels:int,
        kernel_size:Sequence[Union[Sequence[int], int]],
        strides:Sequence[Union[Sequence[int], int]],
        upsample_kernel_size:Sequence[Union[Sequence[int], int]],
        filters:Optional[Sequence[int]] = None,
        activation:str = "LRELU",
        normalisation:str = "instance",
        deep_supervision:bool = False):

        super().__init__()

        if normalisation == "batch":
            Norm = nn.BatchNorm3d
        elif normalisation == "instance":
            Norm = nn.InstanceNorm3d
        else:
            raise KeyError(f"please provide batch or instance for normalisation, not {normalisation}")

        if activation == "PRELU":
            Act = nn.PReLU
        elif activation == "LRELU":
            Act = nn.LeakyReLU
        elif activation == "RELU":
            Act = nn.ReLU
        else:
            raise KeyError(f"please provide batch or instance for normalisation, not {normalisation}")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        if filters == None:
            self.filters = (32, 64, 128, 256, 320, 320, 320, 320)[:len(self.kernel_size)]
        else:
            self.filters = filters
        self.upsample_kernel_size = upsample_kernel_size[::-1]
        self.in_channels = in_channels
        self.deep_supervision = deep_supervision
        self.down = []
        self.up = []
        self.deepsupervision = []
        self.name = "UNet"

        for i, kernel in enumerate(self.kernel_size):
            in_channels = self.in_channels if i == 0 else out_channels
            out_channels = self.filters[i]
            # This is just for clarity in printing the network
            if i == 0:
                self.input_block = DoubleConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=self.strides[i],
                    Norm=Norm,
                    Activation=Act
                    )
            elif i < len(self.kernel_size)-1:
                self.down.append(
                    DoubleConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel,
                        stride=self.strides[i],
                        Norm=Norm,
                        Activation=Act
                        )
                    )
            else:
                self.down = nn.ModuleList(self.down)
                self.bottleneck = DoubleConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=self.strides[i],
                    Norm=Norm,
                    Activation=Act
                    )

        for i, kernel in enumerate(self.kernel_size[::-1][1:]):
            in_channels = out_channels
            out_channels = self.filters[::-1][1:][i]
            self.up.append(
                Up(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    #stride=self.strides[::-1][i],
                    upsample_kernel_size=self.upsample_kernel_size[i],
                    Norm=Norm)
                )

        self.up = nn.ModuleList(self.up)
        self.finalconv = nn.Conv3d(out_channels, self.out_channels, kernel_size=1, stride=1)

        if self.deep_supervision == True:
            # Deep supervision for all but final and two lowest resolutions
            for i, kernel in enumerate(self.kernel_size[::-1][2:-1]):
                out_channels = self.filters[::-1][i+2]
                self.deepsupervision.append(
                        nn.Conv3d(
                            in_channels=out_channels,
                            out_channels=2,
                            kernel_size=1, 
                            stride=1,
                            bias=False)
                        )

            self.deepsupervision = nn.ModuleList(self.deepsupervision)

        # Weight initialization
        self.weight_initializer()

    def weight_initializer(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose3d) or isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, a=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.input_block(x)
        skips = [x]

        for module in self.down:
            x = module(x)
            skips.append(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        supervision = []
        for i, module in enumerate(self.up):
            x_skip = skips[i]
            x = module(x, x_skip)
            if self.deepsupervision and 0 < i < len(self.up)-1:
                x_deep = self.deepsupervision[i-1](x)
                supervision.append(x_deep)

        x = self.finalconv(x)
        
        if supervision:
            supervision.append(x)
            return supervision
        
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, Norm=nn.InstanceNorm3d, Activation=nn.LeakyReLU):
        super(DoubleConv, self).__init__()
        # Checking kernel for padding
        if kernel_size == 3 or kernel_size == [3,3,3]:
            padding = 1
        elif kernel_size == [3,3,1]:
            padding = (1, 1, 0)
        elif kernel_size == [1,3,3]:
            padding = (0, 1, 1)
        else:
            padding = 1
            warnings.warn("kernel is neither 3, (3,3,3) or (1,3,3). This scenario has not been correctly implemented yet, but using padding = 1")

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size,
                    stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size,
                    stride=1, padding=padding, bias=bias)
        self.act = Activation()

        self.norm1 = Norm(out_channels, affine=True)
        self.norm2 = Norm(out_channels, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x

class Up(nn.Module):
    """
    A helper Module that performs 2 convolutions, 1 UpConvolution and a uses a skip connection.
    A PReLU activation and optionally a BatchNorm or InstanceNorm follows each convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, upsample_kernel_size, Norm=nn.InstanceNorm3d):
        super(Up, self).__init__()

        self.transpconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=upsample_kernel_size, stride=upsample_kernel_size, bias=False)   
        self.doubleconv = DoubleConv(out_channels*2, out_channels, kernel_size, stride=1, Norm=Norm)

    def forward(self, x, x_skip):
        x = self.transpconv(x)
        print(x.size())
        print(x_skip.size())
        x = torch.cat((x_skip, x), dim=1)
        x = self.doubleconv(x)
        return x