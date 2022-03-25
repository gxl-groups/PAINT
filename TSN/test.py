# -*- coding: utf-8 -*-

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from PIL import Image
import math
import os
import argparse
import importlib
import datetime
import json

### My libs
from core.utils import set_device, postprocess, ZipReader, set_seed
from core.utils import postprocess
from core.dataset import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="MGP")
parser.add_argument('-c', '--config', type=str, default='configs/celebahq.json')
parser.add_argument("-l", "--level", type=int, required=False, default=None)
parser.add_argument("-n", "--model_name", default='pennet', type=str)
parser.add_argument("-m", "--mask", default='pconv', type=str)
parser.add_argument("-s", "--size", default=256, type=int)
parser.add_argument("-p", "--port", type=str, default="23451")
args = parser.parse_args()

BATCH_SIZE = 6

def mask_c(size):
  mask_c = torch.zeros(1, size * size, size, size).cuda()
  for i in range(size):
    for j in range(size):
      mask_c[:, i * size + j, i, j] = 1
      if j >= 1:
        mask_c[:, i * size + j - 1, i, j] = 1
      if j <= size - 1 -1:
        mask_c[:, i * size + j + 1, i, j] = 1
      if i <= size - 1 -1:
        mask_c[:, i * size + j + size, i, j] = 1
      if i >= 1:
        mask_c[:, i * size + j - size, i, j] = 1
  return mask_c

def main_worker(gpu, ngpus_per_node, config):
    torch.cuda.set_device(gpu)
    set_seed(config['seed'])

    # Model and version
    net = importlib.import_module('model.' + args.model_name)
    model = set_device(net.InpaintGenerator())
    latest_epoch = open(os.path.join(config['save_dir'], 'latest.ckpt'), 'r').read().splitlines()[-1]
    path = os.path.join(config['save_dir'], 'gen_{}.pth'.format(latest_epoch))
    data = torch.load(path, map_location=lambda storage, loc: set_device(storage))
    model.load_state_dict(data['netG'])
    model.eval()

    # prepare dataset
    dataset = Dataset(config['data_loader'], debug=False, split='test', level=args.level)
    step = math.ceil(len(dataset) / ngpus_per_node)
    dataset.set_subset(gpu * step, min(gpu * step + step, len(dataset)))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=config['trainer']['num_workers'],
                            pin_memory=True)

    path = os.path.join(config['save_dir'],
                        'results_{}_level_{}'.format(str(latest_epoch).zfill(5), str(args.level).zfill(2)))
    os.makedirs(path, exist_ok=True)
    mask4 = mask_c(4)
    mask8 = mask_c(8)
    mask16 = mask_c(16)
    mask32 = mask_c(32)
    mask64 = mask_c(64)
    # iteration through datasets
    for idx, (names, parsing, img2, mask_t, mask_b, imgt_masked, imgb_masked, mask_to, mask_bo) in enumerate(
            dataloader):
        print('[{}] GPU{} {}/{}: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            gpu, idx, len(dataloader), names[0]))
        parsing, img2, mask_t, mask_b, imgt_masked, imgb_masked, mask_to, mask_bo = set_device(
            [parsing, img2, mask_t, mask_b, imgt_masked, imgb_masked, mask_to, mask_bo])
        images_masked = imgt_masked + imgb_masked
        with torch.no_grad():
            output = model(imgt_masked, imgb_masked, mask_t, mask_b, mask_to, mask_bo, parsing, mask4, mask8, mask16, mask32, mask64)
        orig_imgs = postprocess(img2)
        mask_imgs = postprocess(images_masked)
        pred_imgs = postprocess(output)
        for i in range(len(orig_imgs)):
            Image.fromarray(pred_imgs[i]).save(os.path.join(path, '{}.png'.format(names[i].split('.')[0])))
            Image.fromarray(orig_imgs[i]).save(os.path.join(path, '{}_orig.png'.format(names[i].split('.')[0])))
            Image.fromarray(mask_imgs[i]).save(os.path.join(path, '{}_mask.png'.format(names[i].split('.')[0])))
    print('Finish in {}'.format(path))


if __name__ == '__main__':
    ngpus_per_node = torch.cuda.device_count()
    config = json.load(open(args.config))
    if args.mask is not None:
        config['data_loader']['mask'] = args.mask
    if args.size is not None:
        config['data_loader']['w'] = config['data_loader']['h'] = args.size
    config['model_name'] = args.model_name
    config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}'.format(config['model_name'],
                                                                              config['data_loader']['name'],
                                                                              config['data_loader']['mask'],
                                                                              config['data_loader']['w']))

    print('using {} GPUs for testing ... '.format(ngpus_per_node))
    # setup distributed parallel training environments
    ngpus_per_node = torch.cuda.device_count()
    config['world_size'] = ngpus_per_node
    config['init_method'] = 'tcp://127.0.0.1:' + args.port
    config['distributed'] = True
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))