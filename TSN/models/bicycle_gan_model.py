import torch
from .base_model import BaseModel
from . import networks
from .utils import generate_label
import cv2


class BiCycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        if opt.isTrain:
            assert opt.batch_size % 2 == 0  # load two images at one time.

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'G_L1', 'kl']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A_encoded_vis', 'real_b_encoded', 'fake_B_random', 'fake_B_encoded']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        use_E = opt.isTrain or not opt.no_encode
        use_vae = True
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)
        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_D2:
            self.model_names += ['D2']
            self.netD2 = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                           init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        else:
            self.netD2 = None
        if use_E:
            self.model_names += ['E']
            self.netE = networks.define_E(4, 8, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, vaeLike=use_vae)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionZ = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D2:
                self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D2)

    def is_train(self):
        """check if the current batch is good for training."""
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_ct = input['ct' if AtoB else 'A'].to(self.device)
        self.real_cb = input['cb' if AtoB else 'A'].to(self.device)
        self.real_c3 = input['c3' if AtoB else 'A'].to(self.device)
        self.real_c4 = input['c4' if AtoB else 'A'].to(self.device)
        self.real_c5 = input['c5' if AtoB else 'A'].to(self.device)
        self.sketch = input['sketch' if AtoB else 'A'].to(self.device)
        self.A_vis = input['A_vis' if AtoB else 'B_vis'].to(self.device)
        self.B_vis = input['B_vis' if AtoB else 'A_vis'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, 8)
            z1 = torch.randn(batch_size, 8)
            z2 = torch.randn(batch_size, 8)
            # # # # z3 = torch.randn(batch_size, 8)
            # # # # z4 = torch.randn(batch_size, 8)
            z = torch.cat((z, z1, z2), 1)
        return z.to(self.device)

    def encode(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def test(self, z0=None, encode=False):
        with torch.no_grad():
            if encode:  # use encoded z
                z0t, _ = self.netE(self.real_ct)
                z0b, _ = self.netE(self.real_cb)
                z03, _ = self.netE(self.real_c3)
                z0 = torch.cat((z0t, z0b, z03), 1)
            if z0 is None:
                z0t = self.get_z_random(self.real_A.size(0), 4)
                z0b = self.get_z_random(self.real_A.size(0), 4)
                z0 = torch.cat((z0t, z0b), 1)
            self.fake_B = self.netG(self.real_A, z0)
            return self.real_A, self.fake_B, self.real_B, self.A_vis

    def forward(self):
        # get real images
        half_size = self.opt.batch_size // 2
        # A1, B1 for encoded; A2, B2 for random
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_b_encoded = self.B_vis[0:half_size]
        self.real_A_encoded_vis = self.A_vis[0:half_size]
        self.real_B_encoded = self.real_B[0:half_size]
        self.real_B_random = self.real_B[half_size:]
        self.real_B_ct = self.real_ct[0:half_size]
        self.real_B_cb = self.real_cb[0:half_size]
        self.real_B_c3 = self.real_c3[0:half_size]
        self.real_B_c4 = self.real_c4[0:half_size]
        self.real_B_c5 = self.real_c5[0:half_size]
        self.sketch = self.sketch[0:half_size]
        # get encoded z
        self.z_encoded_t, self.mut, self.logvart = self.encode(self.real_B_ct)
        self.z_encoded_b, self.mub, self.logvarb = self.encode(self.real_B_cb)
        self.z_encoded_3, self.mu3, self.logvar3 = self.encode(self.real_B_c3)
        # self.z_encoded_4, self.mu4, self.logvar4 = self.encode(self.real_B_c4)
        # self.z_encoded_5, self.mu5, self.logvar5 = self.encode(self.real_B_c5)
        # get random z
        self.z_encoded = torch.cat((self.z_encoded_t, self.z_encoded_b, self.z_encoded_3), 1)
        self.z_random_t = self.get_z_random(self.real_A_encoded.size(0), 8)
        self.z_random_b = self.get_z_random(self.real_A_encoded.size(0), 8)
        self.z_random_3 = self.get_z_random(self.real_A_encoded.size(0), 8)
        # self.z_random_4 = self.get_z_random(self.real_A_encoded.size(0), 8)
        # self.z_random_5 = self.get_z_random(self.real_A_encoded.size(0), 8)
        self.z_random = torch.cat((self.z_random_t, self.z_random_b, self.z_random_3), 1)
        # generate fake_B_encoded
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)
        # generate fake_B_random
        self.fake_B_random = self.netG(self.real_A_encoded, self.z_random)
        if self.opt.conditional_D:   # tedious conditoinal data
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_b_encoded], 1)
            self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            self.real_data_random = torch.cat([self.real_A[half_size:], self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.fake_data_random = self.fake_B_random
            self.real_data_encoded = self.real_b_encoded
            self.real_data_random = self.real_B_random

        # compute z_predict
        parsing_all = self.fake_B_random[0].argmax(dim=0)
        if self.opt.lambda_z > 0.0:
            parsing_im_b_201 = torch.zeros(1, 3, 256, 256).cuda()
            for i in range(0, 3):
                parsing_im_b_201[:, i, :, :] += (parsing_all == i + 5)
            parsing_im_b = torch.zeros(256, 256).cuda()
            for i in range(3):
                parsing_im_b = parsing_im_b + parsing_im_b_201[0 , i, :, :]
            parsing_im_b = parsing_im_b.cpu().numpy()
            parsing_im_b = parsing_im_b.astype("uint8")
            sketch201 = cv2.Canny(parsing_im_b, 0, 1)
            sketch201 = torch.from_numpy(sketch201)
            sketch201 = sketch201.unsqueeze(dim=0).unsqueeze(dim=1).float().cuda() / 255
            parsing_im_b_201 = torch.cat((parsing_im_b_201, sketch201), 1)

            parsing_im_b_202 = torch.zeros(1, 3, 256, 256).cuda()
            parsing_im_b_202[:, 0, :, :] += (parsing_all == 8)
            parsing_im_b_202[:, 1, :, :] += (parsing_all == 9)
            parsing_im_b_202[:, 2, :, :] += (parsing_all == 12)
            parsing_im_b = torch.zeros(256, 256).cuda()
            for i in range(3):
                parsing_im_b = parsing_im_b + parsing_im_b_202[0, i, :, :]
            parsing_im_b = parsing_im_b.cpu().numpy()
            parsing_im_b = parsing_im_b.astype("uint8")
            sketch202 = cv2.Canny(parsing_im_b, 0, 1)
            sketch202 = torch.from_numpy(sketch202)
            # sketch202 = self.sketch * parsing_im_b
            sketch202 = sketch202.unsqueeze(dim=0).unsqueeze(dim=1).float().cuda() / 255
            parsing_im_b_202 = torch.cat((parsing_im_b_202, sketch202), 1)

            parsing_im_b_203 = torch.zeros(1, 3, 256, 256).cuda()
            parsing_im_b_203[:, 0, :, :] += (parsing_all == 1)
            parsing_im_b_203[:, 1, :, :] += (parsing_all == 2)
            parsing_im_b_203[:, 2, :, :] += (parsing_all == 13)
            parsing_im_b = torch.zeros(256, 256).cuda()
            for i in range(3):
                parsing_im_b = parsing_im_b + parsing_im_b_203[0, i, :, :]
            parsing_im_b = parsing_im_b.cpu().numpy()
            parsing_im_b = parsing_im_b.astype("uint8")
            sketch203 = cv2.Canny(parsing_im_b, 0, 1)
            sketch203 = torch.from_numpy(sketch203)
            # sketch203 = self.sketch * parsing_im_b
            sketch203 = sketch203.unsqueeze(dim=0).unsqueeze(dim=1).float().cuda() / 255
            parsing_im_b_203 = torch.cat((parsing_im_b_203, sketch203), 1)

            parsing_im_b_204 = torch.zeros(1, 3, 256, 256).cuda()
            parsing_im_b_204[:, 0, :, :] += (parsing_all == 10)
            parsing_im_b_204[:, 1, :, :] += (parsing_all == 14)
            parsing_im_b_204[:, 2, :, :] += (parsing_all == 15)
            # parsing_im_b = torch.zeros(1, 1, 256, 256).cuda()
            # for i in range(3):
            #     parsing_im_b = parsing_im_b + parsing_im_b_204[:, i, :, :]
            # sketch204 = self.sketch * parsing_im_b
            # parsing_im_b_204 = torch.cat((parsing_im_b_204, sketch204), 1)

            parsing_im_b_205 = torch.zeros(1, 3, 256, 256).cuda()
            parsing_im_b_205[:, 0, :, :] += (parsing_all == 11)
            parsing_im_b_205[:, 1, :, :] += (parsing_all == 16)
            parsing_im_b_205[:, 2, :, :] += (parsing_all == 17)
            # parsing_im_b = torch.zeros(1, 1, 256, 256).cuda()
            # for i in range(3):
            #     parsing_im_b = parsing_im_b + parsing_im_b_205[:, i, :, :]
            # sketch205 = self.sketch * parsing_im_b
            # parsing_im_b_205 = torch.cat((parsing_im_b_205, sketch205), 1)

            self.mut2, logvart2 = self.netE(parsing_im_b_201)  # mu2 is a point estimate
            self.mub2, logvarb2 = self.netE(parsing_im_b_202)
            self.mu3, logvar3 = self.netE(parsing_im_b_203)
            # self.mu4, logvar4 = self.netE(parsing_im_b_204)
            # self.mu5, logvar5 = self.netE(parsing_im_b_205)


    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach())
        # real
        pred_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = (loss_D_fake + loss_D_real)
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, self.opt.lambda_GAN)
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.opt.lambda_GAN2)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.opt.lambda_GAN2)
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            self.loss_kl = torch.sum(1 + self.logvart - self.mut.pow(2) - self.logvart.exp()) * (-0.5 * self.opt.lambda_kl)
            self.loss_kl += torch.sum(1 + self.logvarb - self.mub.pow(2) - self.logvarb.exp()) * (-0.5 * self.opt.lambda_kl)
            self.loss_kl += torch.sum(1 + self.logvar3 - self.mu3.pow(2) - self.logvar3.exp()) * (
                        -0.5 * self.opt.lambda_kl)
            # self.loss_kl += torch.sum(1 + self.logvar4 - self.mu4.pow(2) - self.logvar4.exp()) * (
            #             -0.5 * self.opt.lambda_kl)
            # self.loss_kl += torch.sum(1 + self.logvar5 - self.mu5.pow(2) - self.logvar5.exp()) * (
            #             -0.5 * self.opt.lambda_kl)
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            # self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
            self.loss_G_L1 = self.cross_entropy(self.fake_B_encoded, self.real_b_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad([self.netD, self.netD2], True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random)
            self.optimizer_D2.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = torch.mean(torch.abs(self.mut2 - self.z_random_t)) * self.opt.lambda_z
            self.loss_z_L1 += torch.mean(torch.abs(self.mub2 - self.z_random_b)) * self.opt.lambda_z
            self.loss_z_L1 += torch.mean(torch.abs(self.mu3 - self.z_random_3)) * self.opt.lambda_z
            # self.loss_z_L1 += torch.mean(torch.abs(self.mu4 - self.z_random_4)) * self.opt.lambda_z
            # self.loss_z_L1 += torch.mean(torch.abs(self.mu5 - self.z_random_5)) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad([self.netD, self.netD2], False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()
        self.optimizer_E.step()
        # update G only
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()

    def cross_entropy(self, fake_out, real_parsing):
        # 取log
        fake_out = fake_out + 0.00001
        log_output = torch.log(fake_out)
        loss = - torch.sum((real_parsing * log_output))
        loss = loss / (256*256)
        return loss