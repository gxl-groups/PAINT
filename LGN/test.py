import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
from utils import generate_label
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from miou import *

# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)

def save_image2(image_numpy, image_path):
    img = Image.fromarray(np.uint8(image_numpy))
    img.save(image_path)

# test stage
sum = 0
for i, data in enumerate(islice(dataset, opt.num_test)):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    # print('---------------------z', z_samples[[0]][0])
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        # z_samples[[0]][0][0] = z_samples[[0]][0][0] * nn / 10
        # print('-------------------z1', z_samples[[0]][0][0] * nn)
        # print('-------------------z', z_samples[[0]])
        z = z_samples[[nn]].clone()
        # z[:, 0: 8] = z_samples[[0]][:, 0: 8]
        # z[:, 8 : 16] = z_samples[[0]][:, 8 : 16]
        # z[:, 16: 24] = z_samples[[0]][:, 16: 24]
        # z[0][0] = z_samples[[0]][0][0] * nn / 9
        # z[0][1] = z_samples[[0]][0][1] * nn / 10
        real_A, fake_B, real_B, A_vis = model.test(z, encode=encode)
        # fake_B_vis = generate_label(fake_B, 256, 256)
        # real_B_vis = generate_label(real_B, 256, 256)
        # A_vis = A_vis.numpy()
        # fake_B_vis = fake_B_vis.numpy()
        # real_B_vis = real_B_vis.numpy()
        path = '/data/hj/Projects/BicycleGAN/results/new_25-40_latest_val/'
        if nn == 0:
            images = [A_vis, generate_label(real_B, 256, 256), generate_label(fake_B, 256, 256)]
            names = ['input', 'ground truth', 'encoded']
            real_B_copy = real_B.max(1)[1]
            fake_B_copy = fake_B.max(1)[1]
            save_image2(real_B_copy[0].cpu().numpy(), path + 'RB/' + str(i) + '.png')
            save_image2(fake_B_copy[0].cpu().numpy(), path + 'FB/' + str(i) + '.png')
            save_image(generate_label(real_B, 256, 256), path + 'ground truth/' + str(i) + '.png', normalize=True)
            save_image(generate_label(fake_B, 256, 256), path + 'encoded/' + str(i) + '.png', normalize=True)
            metric = SegmentationMetric(20)
            ignore_label = [255]
            hist = metric.addBatch(fake_B_copy.cpu(), real_B_copy.cpu(), [255])
            sum = sum + metric.meanIntersectionOverUnion()
            # sum = sum + iou_mean(fake_B_copy, real_B_copy, 19)
        else:
            save_image(generate_label(fake_B, 256, 256), path + str(nn) + '/' + str(i) + '.png', normalize=True)

print('---------------------------------all', sum / 1999)

webpage.save()

