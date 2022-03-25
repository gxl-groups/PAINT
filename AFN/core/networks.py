import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np


###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def affine_transformation(X, alpha, beta):
    x = X.clone()
    mean, std = calc_mean_std(x)
    mean = mean.expand_as(x)
    std = std.expand_as(x)
    return alpha * ((x - mean) / std) + beta


###############################################################################
# Defining G/D
###############################################################################

def define_G(input_nc, guide_nc, output_nc, ngf, netG, n_layers=8, n_downsampling=3, n_blocks=9, norm='batch',
             init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'bFT_resnet':
        net = bFT_Resnet(input_nc, guide_nc, output_nc, ngf, norm_layer=norm_layer, n_blocks=n_blocks)
    elif netG == 'bFT_unet':
        net = bFT_Unet(input_nc, guide_nc, output_nc, n_layers, ngf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    net = init_net(net, init_type, init_gain, gpu_ids)

    return net


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02,
             num_D=1, getIntermFeat=False, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D2(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real).cuda()
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real).cuda()
            return self.loss(input[-1], target_tensor)


class GANLoss2(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss2, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True)):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation)

    def build_conv_block(self, dim, padding_type, norm_layer, activation):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


##############################################################################
# Discriminators
##############################################################################

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
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
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator2(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


##############################################################################
# Generators
##############################################################################

class bFT_Unet(nn.Module):
    def __init__(self, input_nc, guide_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
                 bottleneck_depth=100):
        super(bFT_Unet, self).__init__()

        self.num_downs = num_downs

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.downconv1 = nn.Sequential(*[nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.downconv2 = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.downconv3 = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.downconv4 = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)])

        downconv = []  ## this has #(num_downs - 5) layers each with [relu-downconv-norm]
        for i in range(num_downs - 5):
            downconv += [nn.LeakyReLU(0.2, True),
                         nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        self.downconv = nn.Sequential(*downconv)
        self.downconv5 = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)])

        ### bottleneck ------

        self.upconv1 = nn.Sequential(
            *[nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
              norm_layer(ngf * 8)])
        upconv = []  ## this has #(num_downs - 5) layers each with [relu-upconv-norm]
        for i in range(num_downs - 5):
            upconv += [nn.ReLU(True),
                       nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * 8)]
        self.upconv = nn.Sequential(*upconv)
        self.upconv2 = nn.Sequential(*[nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1,
                                                          bias=use_bias), norm_layer(ngf * 4)])
        self.upconv3 = nn.Sequential(*[nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1,
                                                          bias=use_bias), norm_layer(ngf * 2)])
        self.upconv4 = nn.Sequential(
            *[nn.ReLU(True), nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
              norm_layer(ngf)])
        self.upconv5 = nn.Sequential(
            *[nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1), nn.Tanh()])

        ### guide downsampling
        self.G_downconv1 = nn.Sequential(*[# nn.Conv2d(guide_nc, guide_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                           # nn.Conv2d(guide_nc, guide_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                           # nn.Conv2d(guide_nc, guide_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                           # nn.Conv2d(guide_nc, guide_nc, kernel_size=3, stride=1, padding=1,
                                           #           bias=use_bias),
                                           # nn.Conv2d(guide_nc, guide_nc, kernel_size=3, stride=1, padding=1,
                                           #           bias=use_bias),
                                           # nn.Conv2d(guide_nc, guide_nc, kernel_size=3, stride=1, padding=1,
                                           #           bias=use_bias),
                                           # nn.Conv2d(guide_nc, guide_nc, kernel_size=3, stride=1, padding=1,
                                           #           bias=use_bias),
                                           # nn.Conv2d(guide_nc, guide_nc, kernel_size=3, stride=1, padding=1,
                                           #           bias=use_bias),
                                           nn.Conv2d(guide_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.G_downconv2 = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.G_downconv3 = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        self.G_downconv4 = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)])
        G_downconv = []  ## this has #(num_downs - 5) layers each with [relu-downconv-norm]
        for i in range(num_downs - 5):
            G_downconv += [nn.LeakyReLU(0.2, True),
                           nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        self.G_downconv = nn.Sequential(*G_downconv)

        ### bottlenecks for param generation
        self.bottleneck_alpha_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.bottleneck_beta_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.bottleneck_alpha_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.bottleneck_beta_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.bottleneck_alpha_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        self.bottleneck_beta_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        bottleneck_alpha = []
        bottleneck_beta = []
        for i in range(num_downs - 5):
            bottleneck_alpha += self.bottleneck_layer(ngf * 8, bottleneck_depth)
            bottleneck_beta += self.bottleneck_layer(ngf * 8, bottleneck_depth)
        self.bottleneck_alpha = nn.Sequential(*bottleneck_alpha)
        self.bottleneck_beta = nn.Sequential(*bottleneck_beta)
        ### for guide
        self.G_bottleneck_alpha_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.G_bottleneck_beta_2 = nn.Sequential(*self.bottleneck_layer(ngf * 2, bottleneck_depth))
        self.G_bottleneck_alpha_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.G_bottleneck_beta_3 = nn.Sequential(*self.bottleneck_layer(ngf * 4, bottleneck_depth))
        self.G_bottleneck_alpha_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        self.G_bottleneck_beta_4 = nn.Sequential(*self.bottleneck_layer(ngf * 8, bottleneck_depth))
        G_bottleneck_alpha = []
        G_bottleneck_beta = []
        for i in range(num_downs - 5):
            G_bottleneck_alpha += self.bottleneck_layer(ngf * 8, bottleneck_depth)
            G_bottleneck_beta += self.bottleneck_layer(ngf * 8, bottleneck_depth)
        self.G_bottleneck_alpha = nn.Sequential(*G_bottleneck_alpha)
        self.G_bottleneck_beta = nn.Sequential(*G_bottleneck_beta)

    def bottleneck_layer(self, nc, bottleneck_depth):
        return [nn.Conv2d(nc, bottleneck_depth, kernel_size=1), nn.ReLU(True),
                nn.Conv2d(bottleneck_depth, nc, kernel_size=1)]

    # per pixel
    def get_FiLM_param_(self, X, i, guide=False):
        x = X.clone()
        # bottleneck
        if guide:
            if (i == '2'):
                alpha_layer = self.G_bottleneck_alpha_2
                beta_layer = self.G_bottleneck_beta_2
            elif (i == '3'):
                alpha_layer = self.G_bottleneck_alpha_3
                beta_layer = self.G_bottleneck_beta_3
            elif (i == '4'):
                alpha_layer = self.G_bottleneck_alpha_4
                beta_layer = self.G_bottleneck_beta_4
            else:  # a number i will be given to specify which bottleneck to use
                alpha_layer = self.G_bottleneck_alpha[i:i + 3]
                beta_layer = self.G_bottleneck_beta[i:i + 3]
        else:
            if (i == '2'):
                alpha_layer = self.bottleneck_alpha_2
                beta_layer = self.bottleneck_beta_2
            elif (i == '3'):
                alpha_layer = self.bottleneck_alpha_3
                beta_layer = self.bottleneck_beta_3
            elif (i == '4'):
                alpha_layer = self.bottleneck_alpha_4
                beta_layer = self.bottleneck_beta_4
            else:  # a number i will be given to specify which bottleneck to use
                alpha_layer = self.bottleneck_alpha[i:i + 3]
                beta_layer = self.bottleneck_beta[i:i + 3]

        alpha = alpha_layer(x)
        beta = beta_layer(x)
        return alpha, beta

    def forward(self, input, guide):
        ## downconv
        down1 = self.downconv1(input)
        G_down1 = self.G_downconv1(guide)

        down2 = self.downconv2(down1)
        G_down2 = self.G_downconv2(G_down1)

        g_alpha2, g_beta2 = self.get_FiLM_param_(G_down2, '2', guide=True)
        i_alpha2, i_beta2 = self.get_FiLM_param_(down2, '2')
        down2 = affine_transformation(down2, g_alpha2, g_beta2)
        G_down2 = affine_transformation(G_down2, i_alpha2, i_beta2)

        down3 = self.downconv3(down2)
        G_down3 = self.G_downconv3(G_down2)

        g_alpha3, g_beta3 = self.get_FiLM_param_(G_down3, '3', guide=True)
        i_alpha3, i_beta3 = self.get_FiLM_param_(down3, '3')
        down3 = affine_transformation(down3, g_alpha3, g_beta3)
        G_down3 = affine_transformation(G_down3, i_alpha3, i_beta3)

        down4 = self.downconv4(down3)
        G_down4 = self.G_downconv4(G_down3)

        g_alpha4, g_beta4 = self.get_FiLM_param_(G_down4, '4', guide=True)
        i_alpha4, i_beta4 = self.get_FiLM_param_(down4, '4')
        down4 = affine_transformation(down4, g_alpha4, g_beta4)
        G_down4 = affine_transformation(G_down4, i_alpha4, i_beta4)

        ## (num_downs - 5) layers
        down = []
        G_down = []
        for i in range(self.num_downs - 5):
            layer = 2 * i
            bottleneck_layer = 3 * i
            downconv = self.downconv[layer:layer + 2]
            G_downconv = self.G_downconv[layer:layer + 2]
            if (layer == 0):
                down += [downconv(down4)]
                G_down += [G_downconv(G_down4)]
            else:
                down += [downconv(down[i - 1])]
                G_down += [G_downconv(G_down[i - 1])]

            g_alpha, g_beta = self.get_FiLM_param_(G_down[i], bottleneck_layer, guide=True)
            i_alpha, i_beta = self.get_FiLM_param_(down[i], bottleneck_layer)
            down[i] = affine_transformation(down[i], g_alpha, g_beta)
            G_down[i] = affine_transformation(G_down[i], i_alpha, i_beta)

        down5 = self.downconv5(down[-1])

        ## concat and upconv
        up = self.upconv1(down5)
        num_down = self.num_downs - 5
        for i in range(self.num_downs - 5):
            layer = 3 * i
            upconv = self.upconv[layer:layer + 3]
            num_down -= 1
            up = upconv(torch.cat([down[num_down], up], 1))
        up = self.upconv2(torch.cat([down4, up], 1))
        up = self.upconv3(torch.cat([down3, up], 1))
        up = self.upconv4(torch.cat([down2, up], 1))
        up = self.upconv5(torch.cat([down1, up], 1))

        return up


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class bFT_Resnet(nn.Module):
    def __init__(self, input_nc, guide_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', bottleneck_depth=100):
        super(bFT_Resnet, self).__init__()

        self.activation = nn.ReLU(True)

        n_downsampling = 3

        ## input
        padding_in = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0)]
        self.padding_in = nn.Sequential(*padding_in)
        self.conv1 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1)

        ## guide
        padding_g = [nn.ReflectionPad2d(3), nn.Conv2d(guide_nc, ngf, kernel_size=7, padding=0)]
        self.padding_g = nn.Sequential(*padding_g)
        self.conv1_g = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1)
        self.conv2_g = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1)
        self.conv3_g = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)
        self.conv4_g = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1)

        # bottleneck1
        self.bottleneck_alpha_1 = self.bottleneck_layer(ngf, bottleneck_depth)
        self.G_bottleneck_alpha_1 = self.bottleneck_layer(ngf, bottleneck_depth)
        self.bottleneck_beta_1 = self.bottleneck_layer(ngf, bottleneck_depth)
        self.G_bottleneck_beta_1 = self.bottleneck_layer(ngf, bottleneck_depth)
        # bottleneck2
        self.bottleneck_alpha_2 = self.bottleneck_layer(ngf * 2, bottleneck_depth)
        self.G_bottleneck_alpha_2 = self.bottleneck_layer(ngf * 2, bottleneck_depth)
        self.bottleneck_beta_2 = self.bottleneck_layer(ngf * 2, bottleneck_depth)
        self.G_bottleneck_beta_2 = self.bottleneck_layer(ngf * 2, bottleneck_depth)
        # bottleneck3
        self.bottleneck_alpha_3 = self.bottleneck_layer(ngf * 4, bottleneck_depth)
        self.G_bottleneck_alpha_3 = self.bottleneck_layer(ngf * 4, bottleneck_depth)
        self.bottleneck_beta_3 = self.bottleneck_layer(ngf * 4, bottleneck_depth)
        self.G_bottleneck_beta_3 = self.bottleneck_layer(ngf * 4, bottleneck_depth)
        # bottleneck4
        self.bottleneck_alpha_4 = self.bottleneck_layer(ngf * 8, bottleneck_depth)
        self.G_bottleneck_alpha_4 = self.bottleneck_layer(ngf * 8, bottleneck_depth)
        self.bottleneck_beta_4 = self.bottleneck_layer(ngf * 8, bottleneck_depth)
        self.G_bottleneck_beta_4 = self.bottleneck_layer(ngf * 8, bottleneck_depth)
        # bottleneck5
        self.bottleneck_alpha_5 = self.bottleneck_layer(ngf * 8, bottleneck_depth)
        self.G_bottleneck_alpha_5 = self.bottleneck_layer(ngf * 8, bottleneck_depth)
        self.bottleneck_beta_5 = self.bottleneck_layer(ngf * 8, bottleneck_depth)
        self.G_bottleneck_beta_5 = self.bottleneck_layer(ngf * 8, bottleneck_depth)

        resnet = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            resnet += [
                ResnetBlock(ngf * mult, padding_type=padding_type, activation=self.activation, norm_layer=norm_layer)]
        self.resnet = nn.Sequential(*resnet)
        decoder = [nn.ConvTranspose2d(ngf * 8, int(ngf * 8), kernel_size=3, stride=2, padding=1, output_padding=1),
                   norm_layer(int(ngf * 8)), self.activation]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                           output_padding=1),
                        norm_layer(int(ngf * mult / 2)), self.activation]
        self.pre_decoder = nn.Sequential(*decoder)
        self.decoder = nn.Sequential(
            *[nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()])

    def bottleneck_layer(self, nc, bottleneck_depth):
        return nn.Sequential(*[nn.Conv2d(nc, bottleneck_depth, kernel_size=1), self.activation,
                               nn.Conv2d(bottleneck_depth, nc, kernel_size=1)])

    def get_FiLM_param_(self, X, i, guide=False):
        x = X.clone()
        # bottleneck
        if guide:
            if (i == 1):
                alpha_layer = self.G_bottleneck_alpha_1
                beta_layer = self.G_bottleneck_beta_1
            elif (i == 2):
                alpha_layer = self.G_bottleneck_alpha_2
                beta_layer = self.G_bottleneck_beta_2
            elif (i == 3):
                alpha_layer = self.G_bottleneck_alpha_3
                beta_layer = self.G_bottleneck_beta_3
            elif (i == 4):
                alpha_layer = self.G_bottleneck_alpha_4
                beta_layer = self.G_bottleneck_beta_4
            elif (i == 5):
                alpha_layer = self.G_bottleneck_alpha_5
                beta_layer = self.G_bottleneck_beta_5
        else:
            if (i == 1):
                alpha_layer = self.bottleneck_alpha_1
                beta_layer = self.bottleneck_beta_1
            elif (i == 2):
                alpha_layer = self.bottleneck_alpha_2
                beta_layer = self.bottleneck_beta_2
            elif (i == 3):
                alpha_layer = self.bottleneck_alpha_3
                beta_layer = self.bottleneck_beta_3
            elif (i == 4):
                alpha_layer = self.bottleneck_alpha_4
                beta_layer = self.bottleneck_beta_4
            elif (i == 5):
                alpha_layer = self.bottleneck_alpha_5
                beta_layer = self.bottleneck_beta_5
        alpha = alpha_layer(x)
        beta = beta_layer(x)
        return alpha, beta

    def forward(self, input, guidance):
        input = self.padding_in(input)
        guidance = self.padding_g(guidance)

        g_alpha1, g_beta1 = self.get_FiLM_param_(guidance, 1, guide=True)
        i_alpha1, i_beta1 = self.get_FiLM_param_(input, 1)
        guidance = affine_transformation(guidance, i_alpha1, i_beta1)
        input = affine_transformation(input, g_alpha1, g_beta1)

        input = self.activation(input)
        guidance = self.activation(guidance)

        input = self.conv1(input)
        guidance = self.conv1_g(guidance)

        g_alpha2, g_beta2 = self.get_FiLM_param_(guidance, 2, guide=True)
        i_alpha2, i_beta2 = self.get_FiLM_param_(input, 2)
        input = affine_transformation(input, g_alpha2, g_beta2)
        guidance = affine_transformation(guidance, i_alpha2, i_beta2)

        input = self.activation(input)
        guidance = self.activation(guidance)

        input = self.conv2(input)
        guidance = self.conv2_g(guidance)

        g_alpha3, g_beta3 = self.get_FiLM_param_(guidance, 3, guide=True)
        i_alpha3, i_beta3 = self.get_FiLM_param_(input, 3)
        input = affine_transformation(input, g_alpha3, g_beta3)
        guidance = affine_transformation(guidance, i_alpha3, i_beta3)

        input = self.activation(input)
        guidance = self.activation(guidance)

        # ------------------后期加的------------------#
        input = self.conv3(input)
        guidance = self.conv3_g(guidance)

        g_alpha4, g_beta4 = self.get_FiLM_param_(guidance, 4, guide=True)
        i_alpha4, i_beta4 = self.get_FiLM_param_(input, 4)
        input = affine_transformation(input, g_alpha4, g_beta4)
        guidance = affine_transformation(guidance, i_alpha4, i_beta4)

        input = self.activation(input)
        guidance = self.activation(guidance)
        # ------------------后期加的------------------#

        input = self.conv4(input)
        guidance = self.conv4_g(guidance)

        g_alpha5, g_beta5 = self.get_FiLM_param_(guidance, 5, guide=True)
        input = affine_transformation(input, g_alpha5, g_beta5)

        input = self.activation(input)

        input = self.resnet(input)
        input = self.pre_decoder(input)
        output = self.decoder(input)
        return output


from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class GramMatrix2(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a, b, c * d)  # resize F_XL into \hat F_XL

        G = torch.bmm(features, features.transpose(1, 2))  # compute the gram product

        # normalize the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(b * c * d)

def GramMatrix(input):
    s = input.size()
    features = input.view(s[0], s[1], s[2]*s[3])
    features_t = torch.transpose(features, 1, 2)
    G = torch.bmm(features, features_t).div(s[1]*s[2]*s[3])
    return G

class StyleLoss(nn.Module):
    def __init__(self, layids = None):
        super(StyleLoss, self).__init__()
        self.vgg = Vgg19()
        # self.gram = gram_matrix()
        self.vgg.cuda()
        # self.gram.cuda()
        self.criterion  = nn.L1Loss()
        self.weight = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weight[i]*self.criterion(GramMatrix(x_vgg[i]),GramMatrix(y_vgg[i]).detach())
        return loss

class FeatureExtractor(nn.Module):
    # Extract features from intermediate layers of a network

    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]