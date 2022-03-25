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

class SPADE(nn.Module):
  def __init__(self, norm_nc, label_nc):
    super().__init__()

    # The dimension of the intermediate embedding space. Yes, hardcoded.
    nhidden = 128
    ks = 3

    pw = ks // 2
    self.x_shared = nn.Sequential(
      nn.Conv2d(norm_nc, norm_nc, kernel_size=1),
    )
    self.mlp_shared = nn.Sequential(
      nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
      nn.ReLU()
    )
    self.mlp_gamma = nn.Sequential(nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw))
    self.mlp_beta = nn.Sequential(nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw))
    self.param_free_norm = nn.InstanceNorm2d(norm_nc)

  def forward(self, x, segmap):

    x = self.x_shared(x)
    x = self.param_free_norm(x)
    # Part 2. produce scaling and bias conditioned on semantic map
    segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
    actv = self.mlp_shared(segmap)
    gamma = self.mlp_gamma(actv)
    beta = self.mlp_beta(actv)

    # apply scale and bias
    out = x * (1 + gamma) + beta

    return out

class SPADEResnetBlock(nn.Module):
  def __init__(self, fin, label_c):
    super().__init__()
    # Attributes
    fout = fin
    self.learned_shortcut = (fin == fout)
    fmiddle = min(fin, fout)

    # create conv layers
    self.conv_0 = nn.Conv2d(fin, fout, kernel_size=3, padding=1)
    self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
    if self.learned_shortcut:
      self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

    # apply spectral norm if specified

    # define normalization layers
    self.norm_0 = SPADE(fin, label_c)
    self.norm_1 = SPADE(fmiddle, label_c)
    if self.learned_shortcut:
      self.norm_s = SPADE(fin, label_c)

  # note the resnet block with SPADE also takes in |seg|,
  # the semantic segmentation map as input
  def forward(self, x, seg):
    # x_s = self.shortcut(x, seg)
    dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
    # dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

    # out = dx + x_s
    out = dx
    out = self.actvn(out)

    return out

  def shortcut(self, x, seg):
    if self.learned_shortcut:
      x_s = self.conv_s(self.actvn(self.norm_s(x, seg)))
    else:
      x_s = x
    return x_s

  def actvn(self, x):
    return F.leaky_relu(x, 2e-1)

class InpaintGenerator(BaseNetwork):
  def __init__(self, init_weights=True, n_res=1, activ='relu',
               norm='none', pad_type='zero', use_sn=False):  # 1046
    super(InpaintGenerator, self).__init__()
    cnum = 32

    self.act = nn.ReLU()

    self.dw_conv01 = PartialConv2d(3, cnum, kernel_size=3, stride=2, padding=1, bias=True, multi_channel=True,
                                   return_mask=True)
    self.bn1 = nn.InstanceNorm2d(cnum)

    self.dw_conv02 = PartialConv2d(cnum, cnum * 2, kernel_size=3, stride=2, padding=1, bias=True, multi_channel=True,
                                   return_mask=True)
    self.bn2 = nn.InstanceNorm2d(cnum * 2)

    self.dw_conv03 = PartialConv2d(cnum * 2, cnum * 4, kernel_size=3, stride=2, padding=1, bias=True,
                                   multi_channel=True,
                                   return_mask=True)
    self.bn3 = nn.InstanceNorm2d(cnum * 4)

    self.dw_conv04 = PartialConv2d(cnum * 4, cnum * 8, kernel_size=3, stride=2, padding=1, bias=True,
                                   multi_channel=True,
                                   return_mask=True)
    self.bn4 = nn.InstanceNorm2d(cnum * 8)

    self.dw_conv05 = PartialConv2d(cnum * 8, cnum * 16, kernel_size=3, stride=2, padding=1, bias=True,
                                   multi_channel=True,
                                   return_mask=True)
    self.bn5 = nn.InstanceNorm2d(cnum * 16)

    self.dw_conv06 = PartialConv2d(cnum * 16, cnum * 16, kernel_size=3, stride=2, padding=1, bias=True,
                                   multi_channel=True,
                                   return_mask=True)
    self.bn6 = nn.InstanceNorm2d(cnum * 16)

    # ------------------------用于下装
    self.bact = nn.ReLU()
    self.bdw_conv01 = PartialConv2d(3, cnum, kernel_size=3, stride=2, padding=1, bias=True, multi_channel=True,
                                    return_mask=True)
    self.bbn1 = nn.InstanceNorm2d(cnum)

    self.bdw_conv02 = PartialConv2d(cnum, cnum * 2, kernel_size=3, stride=2, padding=1, bias=True, multi_channel=True,
                                    return_mask=True)
    self.bbn2 = nn.InstanceNorm2d(cnum * 2)

    self.bdw_conv03 = PartialConv2d(cnum * 2, cnum * 4, kernel_size=3, stride=2, padding=1, bias=True,
                                    multi_channel=True,
                                    return_mask=True)
    self.bbn3 = nn.InstanceNorm2d(cnum * 4)

    self.bdw_conv04 = PartialConv2d(cnum * 4, cnum * 8, kernel_size=3, stride=2, padding=1, bias=True,
                                    multi_channel=True,
                                    return_mask=True)
    self.bbn4 = nn.InstanceNorm2d(cnum * 8)

    self.bdw_conv05 = PartialConv2d(cnum * 8, cnum * 16, kernel_size=3, stride=2, padding=1, bias=True,
                                    multi_channel=True,
                                    return_mask=True)
    self.bbn5 = nn.InstanceNorm2d(cnum * 16)

    self.bdw_conv06 = PartialConv2d(cnum * 16, cnum * 16, kernel_size=3, stride=2, padding=1, bias=True,
                                    multi_channel=True,
                                    return_mask=True)
    self.bbn6 = nn.InstanceNorm2d(cnum * 16)

    # attention module
    self.at_conv05 = AtnConv(cnum * 16, cnum * 16, ksize=1, size=1, fuse=False)  # 4 8
    self.at_conv04 = AtnConv(cnum * 8, cnum * 8, ksize=3, size=2)  # 8 16
    self.at_conv03 = AtnConv(cnum * 4, cnum * 4, ksize=3, size=3)  # 16 32
    self.at_conv02 = AtnConv(cnum * 2, cnum * 2, ksize=3, size=3)  # 32 64
    self.at_conv01 = AtnConv(cnum, cnum, ksize=3, size=3)  # 64 128

    self.bat_conv05 = AtnConv(cnum * 16, cnum * 16, ksize=1, size=1, fuse=False)
    self.bat_conv04 = AtnConv(cnum * 8, cnum * 8, ksize=3, size=2)
    self.bat_conv03 = AtnConv(cnum * 4, cnum * 4, ksize=3, size=3)
    self.bat_conv02 = AtnConv(cnum * 2, cnum * 2, ksize=3, size=3)
    self.bat_conv01 = AtnConv(cnum, cnum, ksize=3, size=3)

    self.spade6 = SPADEResnetBlock(cnum * 16, 20)
    self.spade5 = SPADEResnetBlock(cnum * 16, 20)
    self.spade4 = SPADEResnetBlock(cnum * 8, 20)
    self.spade3 = SPADEResnetBlock(cnum * 4, 20)
    self.spade2 = SPADEResnetBlock(cnum * 2, 20)
    self.spade1 = SPADEResnetBlock(cnum, 20)

    # decoder
    use_bias = True
    self.up_conv05 = nn.Sequential(
      nn.ConvTranspose2d(cnum * 16, cnum * 16, kernel_size=4, stride=2, padding=1, bias=use_bias),
      nn.ReLU()
    )
    self.up_conv04 = nn.Sequential(
      ResBlocks(n_res, cnum * 32, norm, activ, pad_type=pad_type),
      nn.ConvTranspose2d(cnum * 32, cnum * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
      nn.ReLU()
    )
    self.up_conv03 = nn.Sequential(
      ResBlocks(n_res, cnum * 16, norm, activ, pad_type=pad_type),
      nn.ConvTranspose2d(cnum * 16, cnum * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
      nn.ReLU()
    )
    self.up_conv02 = nn.Sequential(
      ResBlocks(n_res, cnum * 8, norm, activ, pad_type=pad_type),
      nn.ConvTranspose2d(cnum * 8, cnum * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
      nn.ReLU()
    )
    self.up_conv01 = nn.Sequential(
      ResBlocks(n_res, cnum * 4, norm, activ, pad_type=pad_type),
      nn.ConvTranspose2d(cnum * 4, cnum, kernel_size=4, stride=2, padding=1, bias=use_bias),
      nn.ReLU()
    )

    self.decoder = nn.Sequential(
      ResBlocks(n_res, cnum * 2, norm, activ, pad_type=pad_type),
      nn.ConvTranspose2d(cnum * 2, 3, kernel_size=4, stride=2, padding=1),
      nn.Tanh()
    )

    if init_weights:
      self.init_weights()

  def bottleneck_layer(self, nc, bottleneck_depth):
    return [nn.Conv2d(nc, bottleneck_depth, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(bottleneck_depth, nc, kernel_size=1)]


  def forward(self, imgt_masked, imgb_masked, mask_t, mask_b, mask_to, mask_bo, parsing):
    x = torch.cat([mask_to, mask_to, mask_to], dim=1)
    # encoder
    x1, mask1 = self.dw_conv01(imgt_masked, x)
    x1 = self.bn1(x1)
    x1 = self.act(x1)
    mask11 = (1 - mask1[:, 0, :, :]).unsqueeze(1)
    x2, mask2 = self.dw_conv02(x1, mask1)
    x2 = self.bn2(x2)
    x2 = self.act(x2)
    mask22 = (1 - mask2[:, 0, :, :]).unsqueeze(1)
    x3, mask3 = self.dw_conv03(x2, mask2)
    x3 = self.bn3(x3)
    x3 = self.act(x3)
    mask33 = (1 - mask3[:, 0, :, :]).unsqueeze(1)
    x4, mask4 = self.dw_conv04(x3, mask3)
    x4 = self.bn4(x4)
    x4 = self.act(x4)
    mask44 = (1 - mask4[:, 0, :, :]).unsqueeze(1)
    x5, mask5 = self.dw_conv05(x4, mask4)
    x5 = self.bn5(x5)
    x5 = self.act(x5)
    mask55 = (1 - mask5[:, 0, :, :]).unsqueeze(1)
    x6, mask6 = self.dw_conv06(x5, mask5)  # 4*4
    xx6 = x6
    x6 = self.bn6(x6)
    x6 = self.act(x6)
    xx6 = self.act(xx6)
    mask66 = (1 - mask6[:, 0, :, :]).unsqueeze(1)

    # attention
    x5 = self.at_conv05(x5, x6, mask55, mask_t)
    x4 = self.at_conv04(x4, x5, mask44, mask_t)
    x3 = self.at_conv03(x3, x4, mask33, mask_t)
    x2 = self.at_conv02(x2, x3, mask22, mask_t)
    x1 = self.at_conv01(x1, x2, mask11, mask_t)

    xb = torch.cat([mask_bo, mask_bo, mask_bo], dim=1)
    # encoder
    xb1, maskb1 = self.bdw_conv01(imgb_masked, xb)
    xb1 = self.bbn1(xb1)
    xb1 = self.bact(xb1)
    maskb11 = (1 - maskb1[:, 0, :, :]).unsqueeze(1)
    xb2, maskb2 = self.bdw_conv02(xb1, maskb1)
    xb2 = self.bbn2(xb2)
    xb2 = self.bact(xb2)
    maskb22 = (1 - maskb2[:, 0, :, :]).unsqueeze(1)
    xb3, maskb3 = self.bdw_conv03(xb2, maskb2)
    xb3 = self.bbn3(xb3)
    xb3 = self.bact(xb3)
    maskb33 = (1 - maskb3[:, 0, :, :]).unsqueeze(1)
    xb4, maskb4 = self.bdw_conv04(xb3, maskb3)
    xb4 = self.bbn4(xb4)
    xb4 = self.bact(xb4)
    maskb44 = (1 - maskb4[:, 0, :, :]).unsqueeze(1)
    xb5, maskb5 = self.bdw_conv05(xb4, maskb4)
    xb5 = self.bbn5(xb5)
    xb5 = self.bact(xb5)
    maskb55 = (1 - maskb5[:, 0, :, :]).unsqueeze(1)
    xb6, maskb6 = self.bdw_conv06(xb5, maskb5)  # 4*4
    xbb6 = xb6
    xb6 = self.bbn6(xb6)
    xb6 = self.bact(xb6)
    xbb6 = self.bact(xbb6)
    maskb66 = (1 - maskb6[:, 0, :, :]).unsqueeze(1)

    # attention
    xb5 = self.bat_conv05(xb5, xb6, maskb55, mask_b)
    xb4 = self.bat_conv04(xb4, xb5, maskb44, mask_b)
    xb3 = self.bat_conv03(xb3, xb4, maskb33, mask_b)
    xb2 = self.bat_conv02(xb2, xb3, maskb22, mask_b)
    xb1 = self.bat_conv01(xb1, xb2, maskb11, mask_b)

    x6 = x6 + xb6
    x5 = x5 + xb5
    x4 = x4 + xb4
    x3 = x3 + xb3
    x2 = x2 + xb2
    x1 = x1 + xb1

    # upx6 = self.spade6(x6, parsing)
    upx5 = self.up_conv05(x6)
    upx5 = self.spade5(upx5, parsing)
    upx4 = self.up_conv04(torch.cat([upx5, x5], dim=1))
    upx4 = self.spade4(upx4, parsing)
    upx3 = self.up_conv03(torch.cat([upx4, x4], dim=1))
    upx3 = self.spade3(upx3, parsing)
    upx2 = self.up_conv02(torch.cat([upx3, x3], dim=1))
    upx2 = self.spade2(upx2, parsing)
    upx1 = self.up_conv01(torch.cat([upx2, x2], dim=1))
    upx1 = self.spade1(upx1, parsing)

    # output
    output = self.decoder(torch.cat([upx1, x1], dim=1))
    return output


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

    model1 = []
    model1 += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]

    self.model = nn.Sequential(*model)
    self.model1 = nn.Sequential(*model1)

  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class Get_image(nn.Module):
  def __init__(self, input_dim, output_dim, activation='tanh'):
    super(Get_image, self).__init__()
    self.conv = Conv2dBlock(input_dim, output_dim, kernel_size=3, stride=1,
                            padding=1, pad_type='reflect', activation=activation)

  def forward(self, x):
    return self.conv(x)

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

class Discriminator(BaseNetwork):
  def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
    super(Discriminator, self).__init__()
    self.getIntermFeat = getIntermFeat
    self.n_layers = n_layers

    kw = 4
    padw = int(np.ceil((kw - 1.0) / 2))
    sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

    nf = ndf
    for n in range(1, n_layers):
      nf_prev = nf
      nf = min(nf * 2, 512)
      sequence += [[
        nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
        norm_layer(nf), nn.LeakyReLU(0.2, True)
      ]]

    nf_prev = nf
    nf = min(nf * 2, 512)
    sequence += [[
      nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
      norm_layer(nf),
      nn.LeakyReLU(0.2, True)
    ]]

    sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

    if use_sigmoid:
      sequence += [[nn.Sigmoid()]]

    if getIntermFeat:
      for n in range(len(sequence)):
        setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
    else:
      sequence_stream = []
      for n in range(len(sequence)):
        sequence_stream += sequence[n]
      self.model = nn.Sequential(*sequence_stream)
    if 1 == 1:
      self.init_weights()

  def forward(self, x):
    return self.model(x)
# #################################################################################
# ########################  Contextual Attention  #################################
# #################################################################################
'''
implementation of attention module
most codes are borrowed from:
1. https://github.com/WonwoongCho/Generative-Inpainting-pytorch/pull/5/commits/9c16537cd123b74453a28cd4e25d3db0505e5881
2. https://github.com/DAA233/generative-inpainting-pytorch/blob/master/model/networks.py
'''
class AtnConv(nn.Module):
  def __init__(self, input_channels=128, output_channels=64, groups=4, ksize=3, stride=1, rate=2, softmax_scale=10.,
               fuse=True, rates=[1, 2, 4, 8], size= 2, use_mask = True):
    super(AtnConv, self).__init__()
    self.ksize = ksize
    self.stride = stride
    self.rate = rate
    self.softmax_scale = softmax_scale
    self.groups = groups
    self.fuse = fuse
    self.mask = use_mask
    self.size = size
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
    kernel = self.size * self.rate
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
      mask = F.interpolate(mask, size=x1s[2:4], mode='bilinear', align_corners=True)
    else:
      mask = torch.zeros([1, 1, x2s[2], x2s[3]])
      if torch.cuda.is_available():
        mask = mask.cuda()
    # extract patches from masks to mask out hole-patches for matching
    m = extract_patches(mask, kernel=kernel, stride=2)
    m = m.contiguous().view(x1s[0], -1, 1, kernel, kernel)  # B*HW*1*K*K
    m = m.mean([2, 3, 4]).unsqueeze(-1).unsqueeze(-1)
    mm = m.eq(0.).float()  # (B, HW, 1, 1)
    mm_groups = torch.split(mm, 1, dim=0)

    y = []
    scale = self.softmax_scale
    # padding = 1
    padding = (kernel - 1) // 2
    padd = (self.ksize - 1) // 2

    # mask_c = torch.zeros(1, x2s[2] // self.stride * x2s[3] // self.stride, x2s[2], x2s[3]).cuda()
    # for i in range(x2s[2]):
    #   for j in range(x2s[3]):
    #     # mask_c[:, i * x2s[3] + j, i, j] = 1
    #     if j >= 1:
    #       mask_c[:, i * x2s[3] + j - 1, i, j] = 1
    #     if j <= x2s[3] - 1 -1:
    #       mask_c[:, i * x2s[3] + j + 1, i, j] = 1
    #     if i <= x2s[2] - 1 -1:
    #       mask_c[:, i * x2s[3] + j + x2s[3], i, j] = 1
    #     if i >= 1:
    #       mask_c[:, i * x2s[3] + j - x2s[3], i, j] = 1

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
      yi = F.conv2d(xi, wi_normed, stride=1, padding=padd)
      yi = yi.contiguous().view(1, x2s[2] // self.stride * x2s[3] // self.stride, x2s[2], x2s[3])
      ma = F.interpolate(ma, size=x2s[2:4], mode='bilinear', align_corners=True)

      # apply softmax to obtain
      yi = yi * mi
      if self.mask:
        yi = yi * ma
      yi = F.softmax(yi * scale, dim=1)
      # yi = yi + yi * mask_c * 0.5
      yi = yi * mi
      if self.mask:
        yi = yi * ma
      yi = yi.clamp(min=1e-8)

      # attending
      wi_center = raw_wi[0]
      # print('-----------------------------------------yi', yi[0][0])
      yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=padding) / float(kernel)
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
    x = nn.ZeroPad2d((kernel - 1) // 2)(x)
  x = x.permute(0, 2, 3, 1)
  all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
  return all_patches


# codes borrowed from https://github.com/DAA233/generative-inpainting-pytorch/blob/master/model/networks.py
def test_contextual_attention(args):
  """Test contextual attention layer with 3-channel image input
  (instead of n-channel feature).
  """
  rate = 2
  stride = 1
  grid = rate * stride
  import cv2
  import numpy as np
  import matplotlib.pyplot as plt

  b = cv2.imread(args[1])
  b = cv2.resize(b, (b.shape[0] // 2, b.shape[1] // 2))
  print(args[1])
  h, w, c = b.shape
  b = b[:h // grid * grid, :w // grid * grid, :]
  b = np.transpose(b, [2, 0, 1])
  b = np.expand_dims(b, 0)
  print('Size of imageA: {}'.format(b.shape))

  f = cv2.imread(args[2])
  h, w, _ = f.shape
  f = f[:h // grid * grid, :w // grid * grid, :]
  f = np.transpose(f, [2, 0, 1])
  f = np.expand_dims(f, 0)
  print('Size of imageB: {}'.format(f.shape))

  bt = torch.Tensor(b)
  ft = torch.Tensor(f)
  atnconv = AtnConv(stride=stride, fuse=False)
  yt = atnconv(ft, bt)
  y = yt.cpu().data.numpy().transpose([0, 2, 3, 1])
  outImg = np.clip(y[0], 0, 255).astype(np.uint8)
  plt.imshow(outImg)
  plt.show()
  print(outImg.shape)
  cv2.imwrite('output.jpg', outImg)


if __name__ == '__main__':
  import sys

  test_contextual_attention(sys.argv)