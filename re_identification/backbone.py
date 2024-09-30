import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import math
import cv2



class MinkEncoderDecoder(nn.Module):
    """
    Basic ResNet architecture using sparse convolutions
    """

    def __init__(self, cfg, template_points):
        super().__init__()

        cr = cfg.CR
        self.D = cfg.DIMENSION
        input_dim = cfg.INPUT_DIM
        self.res = cfg.RESOLUTION
        self.interpolate = False  #cfg.INTERPOLATE
        self.feat_key = cfg.FEAT_KEY

        self.template_points = template_points

        cs = cfg.CHANNELS
        cs = [int(cr*x) for x in cs]
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(input_dim,
                                    cs[0],
                                    kernel_size=3,
                                    stride=1,
                                    dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0],
                                    cs[0],
                                    kernel_size=3,
                                    stride=1,
                                    dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0],
                                  cs[0],
                                  ks=2,
                                  stride=2,
                                  dilation=1,
                                  D=self.D),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1],
                                  cs[1],
                                  ks=2,
                                  stride=2,
                                  dilation=1,
                                  D=self.D),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2],
                                  cs[2],
                                  ks=2,
                                  stride=2,
                                  dilation=1,
                                  D=self.D),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3],
                                  cs[3],
                                  ks=2,
                                  stride=2,
                                  dilation=1,
                                  D=self.D),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )
        self.pool = ME.MinkowskiGlobalAvgPooling()

    def forward(self, in_field):
        x0 = self.stem(in_field.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return torch.cat(self.pool(x4).decomposed_features)


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    stride=stride,
                                    dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(inc,
                                             outc,
                                             kernel_size=ks,
                                             stride=stride,
                                             dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    stride=stride,
                                    dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(outc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    stride=1,
                                    dimension=D),
            ME.MinkowskiBatchNorm(outc),
        )

        self.downsample = (nn.Sequential() if
                           (inc == outc and stride == 1) else nn.Sequential(
                               ME.MinkowskiConvolution(inc,
                                                       outc,
                                                       kernel_size=1,
                                                       dilation=1,
                                                       stride=stride,
                                                       dimension=D),
                               ME.MinkowskiBatchNorm(outc),
                           ))

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

class PositionalEncoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.max_freq = 10000 #cfg.MAX_FREQ
        self.dimensionality = 3 #cfg.DIMENSIONALITY
        self.num_bands = math.floor(feature_dim / self.dimensionality / 2)
        self.base = 2#cfg.BASE
        pad = feature_dim - self.num_bands * 2 * self.dimensionality
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding

    def forward(self, _x):
        """
        _x [B,N,3]: batched point coordinates
        returns: [B,N,C]: positional encoding of dimension C
        """
        x = _x.clone()
        x[:, :, 0] = x[:, :, 0] / 48
        x[:, :, 1] = x[:, :, 1] / 48
        x[:, :, 2] = x[:, :, 2] / 4
        x = x.unsqueeze(-1)
        scales = torch.logspace(
            0.0,
            math.log(self.max_freq / 2) / math.log(self.base),
            self.num_bands,
            base=self.base,
            device=x.device,
            dtype=x.dtype,
        )
        # reshaping
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
        x = x * scales * math.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.flatten(2)
        enc = self.zero_pad(x)
        return enc
