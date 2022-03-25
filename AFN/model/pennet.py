''' Pyramid-Context Encoder Networks: PEN-Net
'''
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from core.spectral_norm import use_spectral_norm
from core.partialconv2d import PartialConv2d
import functools
from core.tools import *
from torch.nn.utils.spectral_norm import spectral_norm



class BaseNetwork(nn.Module):
  def __init__(self):
    super(BaseNetwork, self).__init__()

  def print_network(self):
    if isinstance(self, list):
      self = self[0]
    num_params = 0
    for param in self.parameters():
      num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f million. '
          'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

  def init_weights(self, init_type='normal', gain=0.02):
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    '''

    def init_func(m):
      classname = m.__class__.__name__
      if classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
          nn.init.constant_(m.weight.data, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)
      elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
          nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
          nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
          nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
          nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
          m.reset_parameters()
        else:
          raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)

    self.apply(init_func)

    # propagate to children
    for m in self.children():
      if hasattr(m, 'init_weights'):
        m.init_weights(init_type, gain)


def calc_mean_std(feat, eps=1e-5):
  # eps is a small value added to the variance to avoid divide-by-zero.
  size = feat.size()
  assert (len(size) == 4)
  N, C = size[:2]
  feat_var = feat.view(N, C, -1).var(dim=2) + eps
  feat_std = feat_var.sqrt().view(N, C, 1, 1)
  feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
  return feat_mean, feat_std


class FlowGen(nn.Module):
  def __init__(self, input_dim=3, dim=64, n_res=2, activ='relu',
               norm_flow='ln', norm_conv='in', pad_type='reflect', use_sn=True):
    super(FlowGen, self).__init__()

    self.flow_column = FlowColumn(9, dim, n_res, activ,
                                  norm_flow, pad_type, use_sn)
    self.conv_column = ConvColumn(3, dim, n_res, activ,
                                  norm_conv, pad_type, use_sn)

  def forward(self, parsing, img):
    flow_map32, flow_map64, flow_map128 = self.flow_column(parsing)
    images_out = self.conv_column(img, flow_map32, flow_map64, flow_map128)
    flow_list = [flow_map32, flow_map64, flow_map128]
    return images_out, flow_list

def bilinear_warp(source, flow):
    [b, c, h, w] = source.shape
    x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
    y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
    grid = torch.stack([x,y], dim=0)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    grid = 2*grid - 1
    flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
    grid = (grid+flow).permute(0, 2, 3, 1)
    input_sample = F.grid_sample(source, grid)
    return input_sample

class ConvColumn(nn.Module):
  def __init__(self, input_dim=3, dim=64, n_res=2, activ='lrelu',
               norm='ln', pad_type='reflect', use_sn=True):
    super(ConvColumn, self).__init__()

    self.down_sample = nn.ModuleList()
    self.up_sample = nn.ModuleList()

    # self.down_sample += [nn.Sequential(
    #   Conv2dBlock(input_dim, dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
    #   Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
    #   )]
    #
    # self.down_sample += [nn.Sequential(
    #   Conv2dBlock(dim * 2, dim * 2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
    #   Conv2dBlock(dim * 2, 4 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
    #   )]
    #
    # self.down_sample += [nn.Sequential(
    #   Conv2dBlock(4 * dim, 4 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
    #   Conv2dBlock(4 * dim, 8 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))]
    # dim = 8 * dim

    self.act = nn.ReLU()

    self.dw_conv01 = nn.Conv2d(input_dim, dim // 2, kernel_size=5, stride=1, padding=2, bias=True)
    self.bn1 = nn.InstanceNorm2d(dim // 2)
    self.dw_conv011 = nn.Conv2d(dim // 2, dim, kernel_size=4, stride=2, padding=1, bias=True)
    self.bn11 = nn.InstanceNorm2d(dim)


    self.dw_conv02 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, bias=True)
    self.bn2 = nn.InstanceNorm2d(dim)
    self.dw_conv022 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1, bias=True)
    self.bn22 = nn.InstanceNorm2d(dim * 2)

    self.dw_conv03 = nn.Conv2d(dim * 2, dim * 2, kernel_size=5, stride=1, padding=2, bias=True)
    self.bn3 = nn.InstanceNorm2d(dim * 2)
    self.dw_conv033 = nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1, bias=True)
    self.bn33 = nn.InstanceNorm2d(dim * 4)

    self.dw_conv04 = nn.Conv2d(dim * 4, dim * 4, kernel_size=5, stride=1, padding=2, bias=True)
    self.bn4 = nn.InstanceNorm2d(dim * 4)
    self.dw_conv044 = nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=2, padding=1, bias=True)
    self.bn44 = nn.InstanceNorm2d(dim * 8)
    dim = 8 * dim

    # content decoder
    self.up_sample += [(nn.Sequential(
      ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
      nn.Upsample(scale_factor=2),
      Conv2dBlock(dim, dim // 2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))]

    self.up_sample += [(nn.Sequential(
      ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
      nn.Upsample(scale_factor=2),
      Conv2dBlock(dim, dim // 4, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))]

    self.up_sample += [(nn.Sequential(
      ResBlocks(n_res, dim // 2, norm, activ, pad_type=pad_type),
      nn.Upsample(scale_factor=2),
      Conv2dBlock(dim // 2, dim // 8, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))]

    self.up_sample += [(nn.Sequential(
      ResBlocks(n_res, dim // 4, norm, activ, pad_type=pad_type),
      nn.Upsample(scale_factor=2),
      Conv2dBlock(dim // 4, dim // 8, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
      Get_image(dim // 8, 3)))]

    # self.resample16 = Resample2d(16, 1, sigma=4)
    # self.resample3 = Resample2d(2, 1, sigma=2)
    # self.resample5 = Resample2d(5, 1, sigma=2)

    # self.attn1 = Self_Attn(dim // 2, 'relu')
    # self.attn2 = Self_Attn(dim // 4, 'relu')

    self.at_conv01 = AtnConv(dim // 2, dim // 2)
    self.at_conv02 = AtnConv(dim // 4, dim // 4)

  def forward(self, inputs, flow_map32, flow_map64, flow_map128):
    x10 = self.dw_conv01(inputs)
    x10 = self.bn1(x10)
    x10 = self.act(x10)
    x1 = self.dw_conv011(x10)
    x1 = self.bn11(x1)
    x1 = self.act(x1)

    x20 = self.dw_conv02(x1)
    x20 = self.bn2(x20)
    x20 = self.act(x20)
    x2 = self.dw_conv022(x20)
    x2 = self.bn22(x2)
    x2 = self.act(x2)

    x30 = self.dw_conv03(x2)
    x30 = self.bn3(x30)
    x30 = self.act(x30)
    x3 = self.dw_conv033(x30)
    x3 = self.bn33(x3)
    x3 = self.act(x3)

    x40 = self.dw_conv04(x3)
    x40 = self.bn4(x40)
    x40 = self.act(x40)
    x4 = self.dw_conv044(x40)
    x4 = self.bn44(x4)
    x4 = self.act(x4)

    # x1 = self.down_sample[0](inputs)  # torch.Size([4, 128, 128, 128])
    # x2 = self.down_sample[1](x1)
    # x3 = self.down_sample[2](x2) # torch.Size([4, 512, 32, 32])

    # flow_maps128 = F.interpolate(flow_maps64, size=128, mode='nearest')
    # flow_maps32 = F.interpolate(flow_maps64, scale_factor=1. / 2, mode='nearest')

    flow_fea32 = bilinear_warp(x3, flow_map32)
    flow_fea64 = bilinear_warp(x2, flow_map64)
    # flow_fea32 = x3
    # flow_fea64 = x2
    # flow_fea128 = x1
    flow_fea128 = bilinear_warp(x1, flow_map128)  # torch.Size([4, 256, 64, 64])

    up0 = self.up_sample[0](x4)
    u1 = torch.cat((up0, flow_fea32), 1)
    up1 = self.up_sample[1](u1)
    u1 = torch.cat((up1, flow_fea64), 1)
    up2 = self.up_sample[2](u1)
    u2 = torch.cat((up2, flow_fea128), 1)
    images_out = self.up_sample[3](u2)
    return images_out

  # def resample_image(self, img, flow):
  #   output3 = self.resample5(img, flow)
  #   # output7 = self.resample7(img, flow)
  #   outputs = output3
  #   return outputs

class FlowColumn(nn.Module):
  def __init__(self, input_dim=3, dim=64, n_res=2, activ='lrelu',
               norm='in', pad_type='reflect', use_sn=True):
    super(FlowColumn, self).__init__()

    self.down_sample_flow = nn.ModuleList()
    self.up_sample_flow = nn.ModuleList()

    self.down_sample_flow.append(nn.Sequential(
      Conv2dBlock(input_dim * 2, dim // 2, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn),
      Conv2dBlock(dim // 2, dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
      Conv2dBlock(dim, dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))
    self.down_sample_flow.append(nn.Sequential(
      Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
      Conv2dBlock(2 * dim, 2 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))
    self.down_sample_flow.append(nn.Sequential(
      Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
      Conv2dBlock(4 * dim, 4 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))
    self.down_sample_flow.append(nn.Sequential(
      Conv2dBlock(4 * dim, 8 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
      Conv2dBlock(8 * dim, 8 * dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)))
    dim = 8 * dim

    # content decoder
    self.up_sample_flow.append(nn.Sequential(
      ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
      TransConv2dBlock(dim, dim // 2, 6, 2, 2, norm=norm, activation=activ)))

    self.up_sample_flow.append(nn.Sequential(
      Conv2dBlock(dim, dim // 2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
      ResBlocks(n_res, dim // 2, norm, activ, pad_type=pad_type),
      TransConv2dBlock(dim // 2, dim // 4, 6, 2, 2, norm=norm, activation=activ)))

    self.up_sample_flow.append(nn.Sequential(
      Conv2dBlock(dim // 2, dim // 4, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
      ResBlocks(n_res, dim // 4, norm, activ, pad_type=pad_type),
      TransConv2dBlock(dim // 4, dim // 8, 6, 2, 2, norm=norm, activation=activ)))

    self.location0 = nn.Sequential(
      Conv2dBlock(dim, dim // 4, 5, 1, 2, norm='none', activation='none', pad_type=pad_type, use_sn=use_sn),
      Conv2dBlock(dim // 4, dim // 8, 5, 1, 2, norm='none', activation='none', pad_type=pad_type, use_sn=use_sn),
      Conv2dBlock(dim // 8, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False))

    self.location1 = nn.Sequential(
      Conv2dBlock(dim // 2, dim // 4, 5, 1, 2, norm='none', activation='none', pad_type=pad_type, use_sn=use_sn),
      Conv2dBlock(dim // 4, dim // 8, 5, 1, 2, norm='none', activation='none', pad_type=pad_type, use_sn=use_sn),
      Conv2dBlock(dim // 8, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False))

    self.location2 = nn.Sequential(
      Conv2dBlock(dim // 4, dim // 8, 5, 1, 2, norm='none', activation='none', pad_type=pad_type, use_sn=use_sn),
      Conv2dBlock(dim // 8, dim // 16, 5, 1, 2, norm='none', activation='none', pad_type=pad_type, use_sn=use_sn),
      Conv2dBlock(dim // 16, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False))

    # self.up1 = nn.Sequential(
    #   TransConv2dBlock(2, 2, 6, 2, 2, norm='none', activation='none'))
    #
    # self.up2 = nn.Sequential(
    #   TransConv2dBlock(2, 2, 6, 2, 2, norm='none', activation='none'))

  def forward(self, inputs):
    f_x1 = self.down_sample_flow[0](inputs)
    f_x2 = self.down_sample_flow[1](f_x1)
    f_x3 = self.down_sample_flow[2](f_x2)
    f_x4 = self.down_sample_flow[3](f_x3)

    f_u1 = torch.cat((self.up_sample_flow[0](f_x4), f_x3), 1)
    f_u2 = torch.cat((self.up_sample_flow[1](f_u1), f_x2), 1)
    f_u3 = torch.cat((self.up_sample_flow[2](f_u2), f_x1), 1)
    # print('----------------------------1', f_u3.shape)
    flow_map32 = self.location0(f_u1)
    flow_map64 = self.location1(f_u2)
    flow_map128 = self.location2(f_u3)
    return flow_map32, flow_map64, flow_map128



class Get_image(nn.Module):
  def __init__(self, input_dim, output_dim, activation='tanh'):
    super(Get_image, self).__init__()
    self.conv = Conv2dBlock(input_dim, output_dim, kernel_size=3, stride=1,
                            padding=1, pad_type='reflect', activation=activation)

  def forward(self, x):
    return self.conv(x)


class ResBlocks(nn.Module):
  def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
    super(ResBlocks, self).__init__()
    self.model = []
    for i in range(num_blocks):
      self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]
    self.model = nn.Sequential(*self.model)

  def forward(self, x):
    return self.model(x)


class ResBlock(nn.Module):
  def __init__(self, dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
    super(ResBlock, self).__init__()

    model = []
    model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]
    model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type, use_sn=use_sn)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out


class DilationBlock(nn.Module):
  def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
    super(DilationBlock, self).__init__()

    model = []
    model += [Conv2dBlock(dim, dim, 3, 1, 2, norm=norm, activation=activation, pad_type=pad_type, dilation=2)]
    model += [Conv2dBlock(dim, dim, 3, 1, 4, norm=norm, activation=activation, pad_type=pad_type, dilation=4)]
    model += [Conv2dBlock(dim, dim, 3, 1, 8, norm=norm, activation=activation, pad_type=pad_type, dilation=8)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    out = self.model(x)
    return out

class Conv2dBlock(nn.Module):
  def __init__(self, input_dim, output_dim, kernel_size, stride,
               padding=0, norm='none', activation='relu', pad_type='zero', dilation=1,
               use_bias=True, use_sn=False):
    super(Conv2dBlock, self).__init__()
    self.use_bias = use_bias
    # initialize padding
    if pad_type == 'reflect':
      self.pad = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
      self.pad = nn.ReplicationPad2d(padding)
    elif pad_type == 'zero':
      self.pad = nn.ZeroPad2d(padding)
    else:
      assert 0, "Unsupported padding type: {}".format(pad_type)

    # initialize normalization
    norm_dim = output_dim
    if norm == 'bn':
      self.norm = nn.BatchNorm2d(norm_dim)
    elif norm == 'in':
      self.norm = nn.InstanceNorm2d(norm_dim)
    elif norm == 'ln':
      self.norm = LayerNorm(norm_dim)
    elif norm == 'adain':
      self.norm = AdaptiveInstanceNorm2d(norm_dim)
    elif norm == 'none':
      self.norm = None
    else:
      assert 0, "Unsupported normalization: {}".format(norm)

    # initialize activation
    if activation == 'relu':
      self.activation = nn.ReLU(inplace=True)
    elif activation == 'lrelu':
      self.activation = nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'prelu':
      self.activation = nn.PReLU()
    elif activation == 'selu':
      self.activation = nn.SELU(inplace=True)
    elif activation == 'tanh':
      self.activation = nn.Tanh()
    elif activation == 'sigmoid':
      self.activation = nn.Sigmoid()
    elif activation == 'none':
      self.activation = None
    else:
      assert 0, "Unsupported activation: {}".format(activation)

    # initialize convolution
    if use_sn:
      self.conv = spectral_norm(
        nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=dilation))
    else:
      self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=dilation)

  def forward(self, x):
    x = self.conv(self.pad(x))
    if self.norm:
      x = self.norm(x)
    if self.activation:
      x = self.activation(x)
    return x


class Discriminator(BaseNetwork):
  def __init__(self, in_channels, use_sigmoid=False, use_sn=True, init_weights=True):
    super(Discriminator, self).__init__()
    self.use_sigmoid = use_sigmoid
    cnum = 64
    self.encoder = nn.Sequential(
      use_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=cnum,
                                  kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum, out_channels=cnum * 2,
                                  kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum * 2, out_channels=cnum * 4,
                                  kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum * 4, out_channels=cnum * 8,
                                  kernel_size=5, stride=1, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),
    )

    self.classifier = nn.Conv2d(in_channels=cnum * 8, out_channels=1, kernel_size=5, stride=1, padding=1)
    if init_weights:
      self.init_weights()

  def forward(self, x):
    x = self.encoder(x)
    label_x = self.classifier(x)
    if self.use_sigmoid:
      label_x = torch.sigmoid(label_x)
    return label_x

class TransConv2dBlock(nn.Module):
  def __init__(self, input_dim, output_dim, kernel_size, stride,
               padding=0, norm='none', activation='relu'):
    super(TransConv2dBlock, self).__init__()
    self.use_bias = True

    # initialize normalization
    norm_dim = output_dim
    if norm == 'bn':
      self.norm = nn.BatchNorm2d(norm_dim)
    elif norm == 'in':
      self.norm = nn.InstanceNorm2d(norm_dim)
    elif norm == 'in_affine':
      self.norm = nn.InstanceNorm2d(norm_dim, affine=True)
    elif norm == 'ln':
      self.norm = LayerNorm(norm_dim)
    elif norm == 'adain':
      self.norm = AdaptiveInstanceNorm2d(norm_dim)
    elif norm == 'none':
      self.norm = None
    else:
      assert 0, "Unsupported normalization: {}".format(norm)

    # initialize activation
    if activation == 'relu':
      self.activation = nn.ReLU(inplace=True)
    elif activation == 'lrelu':
      self.activation = nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'prelu':
      self.activation = nn.PReLU()
    elif activation == 'selu':
      self.activation = nn.SELU(inplace=True)
    elif activation == 'tanh':
      self.activation = nn.Tanh()
    elif activation == 'sigmoid':
      self.activation = nn.Sigmoid()
    elif activation == 'none':
      self.activation = None
    else:
      assert 0, "Unsupported activation: {}".format(activation)

    # initialize convolution
    self.transConv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, bias=self.use_bias)

  def forward(self, x):
    x = self.transConv(x)
    if self.norm:
      x = self.norm(x)
    if self.activation:
      x = self.activation(x)
    return x


class AdaptiveInstanceNorm2d(nn.Module):
  def __init__(self, num_features, eps=1e-5, momentum=0.1):
    super(AdaptiveInstanceNorm2d, self).__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    # weight and bias are dynamically assigned
    self.weight = None
    self.bias = None
    # just dummy buffers, not used
    self.register_buffer('running_mean', torch.zeros(num_features))
    self.register_buffer('running_var', torch.ones(num_features))

  def forward(self, x):
    assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
    b, c = x.size(0), x.size(1)
    running_mean = self.running_mean.repeat(b)
    running_var = self.running_var.repeat(b)

    # Apply instance norm
    x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

    out = F.batch_norm(
      x_reshaped, running_mean, running_var, self.weight, self.bias,
      True, self.momentum, self.eps)

    return out.view(b, c, *x.size()[2:])

  def __repr__(self):
    return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine

    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
      self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
    else:
      return F.layer_norm(x, normalized_shape)


class DownsampleResBlock(nn.Module):
  def __init__(self, input_dim, output_dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
    super(DownsampleResBlock, self).__init__()
    self.conv_1 = nn.ModuleList()
    self.conv_2 = nn.ModuleList()

    self.conv_1.append(Conv2dBlock(input_dim, input_dim, 3, 1, 1, 'none', activation, pad_type, use_sn=use_sn))
    self.conv_1.append(Conv2dBlock(input_dim, output_dim, 3, 1, 1, 'none', activation, pad_type, use_sn=use_sn))
    self.conv_1.append(nn.AvgPool2d(kernel_size=2, stride=2))
    self.conv_1 = nn.Sequential(*self.conv_1)

    self.conv_2.append(nn.AvgPool2d(kernel_size=2, stride=2))
    self.conv_2.append(Conv2dBlock(input_dim, output_dim, 1, 1, 0, 'none', activation, pad_type, use_sn=use_sn))
    self.conv_2 = nn.Sequential(*self.conv_2)

  def forward(self, x):
    out = self.conv_1(x) + self.conv_2(x)
    return out

class AtnConv(nn.Module):
  def __init__(self, input_channels=128, output_channels=64, groups=4, ksize=3, stride=1, rate=2, softmax_scale=10.,
               fuse=True, rates=[1, 2, 4, 8]):
    super(AtnConv, self).__init__()
    self.ksize = ksize
    self.stride = stride
    self.rate = rate
    self.softmax_scale = softmax_scale
    self.groups = groups
    self.fuse = fuse
    if self.fuse:
      for i in range(groups):
        self.__setattr__('conv{}'.format(str(i).zfill(2)), nn.Sequential(
          nn.Conv2d(input_channels, output_channels // groups, kernel_size=3, dilation=rates[i], padding=rates[i], bias=True),
          nn.ReLU(inplace=True))
                         )

  def forward(self, x1, x2, mask=None, mask_all=None):
    """ Attention Transfer Network (ATN) is first proposed in
        Learning Pyramid Context-Encoder Networks for High-Quality Image Inpainting. Yanhong Zeng et al. In CVPR 2019.
      inspired by
        Generative Image Inpainting with Contextual Attention, Yu et al. In CVPR 2018.
    Args:
        x1: low-level feature maps with larger resolution.
        x2: high-level feature maps with smaller resolution.
        mask: Input mask, 1 indicates holes.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from b.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.
    Returns:
        torch.Tensor, reconstructed feature map.
    """
    # get shapes
    x1s = list(x1.size())
    x2s = list(x2.size())

    # extract patches from low-level feature maps x1 with stride and rate
    kernel = 2 * self.rate
    raw_w = extract_patches(x1, kernel=kernel, stride=self.rate * self.stride)
    raw_w = raw_w.contiguous().view(x1s[0], -1, x1s[1], kernel, kernel)  # B*HW*C*K*K
    # split tensors by batch dimension; tuple is returned
    raw_w_groups = torch.split(raw_w, 1, dim=0)

    # split high-level feature maps x2 for matching
    f_groups = torch.split(x2, 1, dim=0)
    ma_groups = torch.split(mask_all, 1, dim=0)
    # extract patches from x2 as weights of filter
    w = extract_patches(x2, kernel=self.ksize, stride=self.stride)
    w = w.contiguous().view(x2s[0], -1, x2s[1], self.ksize, self.ksize)  # B*HW*C*K*K
    w_groups = torch.split(w, 1, dim=0)

    # process mask
    if mask is not None:
      mask = F.interpolate(mask, size=x2s[2:4], mode='bilinear', align_corners=True)
    else:
      mask = torch.zeros([1, 1, x2s[2], x2s[3]])
      if torch.cuda.is_available():
        mask = mask.cuda()
    # extract patches from masks to mask out hole-patches for matching
    m = extract_patches(mask, kernel=self.ksize, stride=self.stride)
    m = m.contiguous().view(x2s[0], -1, 1, self.ksize, self.ksize)  # B*HW*1*K*K
    m = m.mean([2, 3, 4]).unsqueeze(-1).unsqueeze(-1)
    mm = m.eq(0.).float()  # (B, HW, 1, 1)
    mm_groups = torch.split(mm, 1, dim=0)

    y = []
    scale = self.softmax_scale
    padding = 0 if self.ksize == 1 else 1

    mask_c = torch.zeros(1, x2s[2] // self.stride * x2s[3] // self.stride, x2s[2], x2s[3]).cuda()
    for i in range(x2s[2]):
      for j in range(x2s[3]):
        # mask_c[:, i * x2s[3] + j, i, j] = 1
        if j >= 1:
          mask_c[:, i * x2s[3] + j - 1, i, j] = 1
        if j <= x2s[3] - 1 -1:
          mask_c[:, i * x2s[3] + j + 1, i, j] = 1
        if i <= x2s[2] - 1 -1:
          mask_c[:, i * x2s[3] + j + x2s[3], i, j] = 1
        if i >= 1:
          mask_c[:, i * x2s[3] + j - x2s[3], i, j] = 1

    for xi, wi, raw_wi, mi, ma in zip(f_groups, w_groups, raw_w_groups, mm_groups, ma_groups):
      '''
      O => output channel as a conv filter
      I => input channel as a conv filter
      xi : separated tensor along batch dimension of front; 
      wi : separated patch tensor along batch dimension of back; 
      raw_wi : separated tensor along batch dimension of back; 
      '''
      # matching based on cosine-similarity
      wi = wi[0]
      escape_NaN = torch.FloatTensor([1e-4])
      if torch.cuda.is_available():
        escape_NaN = escape_NaN.cuda()
      # normalize
      wi_normed = wi / torch.max(torch.sqrt((wi * wi).sum([1, 2, 3], keepdim=True)), escape_NaN)
      yi = F.conv2d(xi, wi_normed, stride=1, padding=padding)
      yi = yi.contiguous().view(1, x2s[2] // self.stride * x2s[3] // self.stride, x2s[2], x2s[3])
      ma = F.interpolate(ma, size=x2s[2:4], mode='bilinear', align_corners=True)

      # apply softmax to obtain
      yi = yi * mi * ma
      yi = F.softmax(yi * scale, dim=1)
      yi = yi + yi * mask_c * 0.5
      yi = yi * mi * ma
      yi = yi.clamp(min=1e-8)

      # attending
      wi_center = raw_wi[0]
      # print('-----------------------------------------yi', yi[0][0])
      yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.
      y.append(yi)
    y = torch.cat(y, dim=0)
    y.contiguous().view(x1s)
    # adjust after filling
    if self.fuse:
      tmp = []
      for i in range(self.groups):
        tmp.append(self.__getattr__('conv{}'.format(str(i).zfill(2)))(y))
      y = torch.cat(tmp, dim=1)
    return y


# extract patches
def extract_patches(x, kernel=3, stride=1):
  if kernel != 1:
    x = nn.ZeroPad2d(1)(x)
  x = x.permute(0, 2, 3, 1)
  all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
  return all_patches


class InpaintGenerator(BaseNetwork):
  def __init__(self, init_weights=True):  # 1046
    super(InpaintGenerator, self).__init__()
    cnum = 32

    # attention module
    self.flow_param = {'input_dim': 3, 'dim': 64, 'n_res': 2, 'activ': 'relu',
                       'norm_conv': 'in', 'norm_flow': 'none', 'pad_type': 'reflect', 'use_sn': False}
    self.f_gen = FlowGen(**self.flow_param)

    if init_weights:
      self.init_weights()


  def forward(self, inputs, parsing, parsingp2):
    # encoder

    outputs, flow = self.f_gen(torch.cat((parsing, parsingp2), dim=1), inputs)

    return outputs, flow


if __name__ == '__main__':
  import sys