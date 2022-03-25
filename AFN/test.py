# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import os
import argparse
import copy
import importlib
import datetime
import random
import sys
import json
import glob

### My libs
from core.utils import set_device, postprocess, ZipReader, set_seed
from core.utils import postprocess
from core.dataset import Dataset
from visualization import generate_label
from utils import flowtoimg

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description="MGP")
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-l", "--level",  type=int, required=False, default=None)
parser.add_argument("-n", "--model_name", type=str, required=True)
parser.add_argument("-m", "--mask", default=None, type=str)
parser.add_argument("-s", "--size", default=None, type=int)
parser.add_argument("-p", "--port", type=str, default="23451")
args = parser.parse_args()

BATCH_SIZE = 1

def main_worker(gpu, ngpus_per_node, config):
  torch.cuda.set_device(gpu)
  set_seed(config['seed'])
  # Parsing Model and version
  pnet = importlib.import_module('model.' + 'parsing_generation')
  pmodel = set_device(pnet.InpaintGenerator())
  latest_epoch = open(os.path.join(config['psave_dir'], 'latest.ckpt'), 'r').read().splitlines()[-1]
  path = os.path.join(config['psave_dir'], 'gen_{}.pth'.format(latest_epoch))
  data = torch.load(path, map_location=lambda storage, loc: set_device(storage))
  pmodel.load_state_dict(data['netG'])
  pmodel.eval()

  # Model and version
  net = importlib.import_module('model.'+args.model_name)
  model = set_device(net.InpaintGenerator())
  latest_epoch = open(os.path.join(config['save_dir'], 'latest.ckpt'), 'r').read().splitlines()[-1]
  path = os.path.join(config['save_dir'], 'gen_{}.pth'.format(latest_epoch))
  data = torch.load(path, map_location = lambda storage, loc: set_device(storage))
  model.load_state_dict(data['netG'])
  model.eval()

  # prepare dataset
  dataset = Dataset(config['data_loader'], debug=False, split='test', level=args.level)
  step = math.ceil(len(dataset) / ngpus_per_node)
  dataset.set_subset(gpu*step, min(gpu*step+step, len(dataset)))
  dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=False, num_workers=config['trainer']['num_workers'], pin_memory=True)

  path = os.path.join(config['save_dir'], 'results_{}_level_{}'.format(str(latest_epoch).zfill(5), str(args.level).zfill(2)))
  os.makedirs(path, exist_ok=True)
  # iteration through datasets
  for idx, (names, parsing, img2, imgp1, parsingp1, imgp2, parsingp2) in enumerate(dataloader):
    print('[{}] GPU{} {}/{}: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      gpu, idx, len(dataloader), names[0]))
    parsing, img2, imgp1, parsingp1, imgp2, parsingp2 = set_device([parsing, img2, imgp1, parsingp1, imgp2, parsingp2])


    with torch.no_grad():
      pred_img1, flow_list1 = model(img2, parsingp1, parsing)
      pred_img2, flow_list2 = model(img2, parsingp2, parsing)

    flow1 = flow_list1[2]
    flow2 = flow_list2[2]

    flow1_vis = flowtoimg(flow1)
    flow2_vis = flowtoimg(flow2)

    orig_imgs = postprocess(img2)
    pose1_imgs = postprocess(imgp1)
    pose2_imgs = postprocess(imgp2)
    pred1_imgs = postprocess(pred_img1)
    pred2_imgs = postprocess(pred_img2)
    parsing_vis = generate_label(parsing, 256, 256)
    fake_parsing2_vis = generate_label(parsingp1, 256, 256)
    fake_parsing3_vis = generate_label(parsingp2, 256, 256)
    p = postprocess(parsing_vis)
    fake_p2 = postprocess(flow1_vis)
    fake_p3 = postprocess(flow2_vis)
    for i in range(len(orig_imgs)):
      Image.fromarray(pred1_imgs[i]).save(os.path.join(path, '{}_pred1.png'.format(names[i].split('.')[0])))
      Image.fromarray(pred2_imgs[i]).save(os.path.join(path, '{}_pred2.png'.format(names[i].split('.')[0])))
      Image.fromarray(orig_imgs[i]).save(os.path.join(path, '{}_orig.png'.format(names[i].split('.')[0])))
      Image.fromarray(pose1_imgs[i]).save(os.path.join(path, '{}_pose1.png'.format(names[i].split('.')[0])))
      Image.fromarray(pose2_imgs[i]).save(os.path.join(path, '{}_pose2.png'.format(names[i].split('.')[0])))
      Image.fromarray(p[i]).save(os.path.join(path, '{}_parsing.png'.format(names[i].split('.')[0])))
      Image.fromarray(fake_p2[i]).save(os.path.join(path, '{}_fp2.png'.format(names[i].split('.')[0])))
      Image.fromarray(fake_p3[i]).save(os.path.join(path, '{}_fp3.png'.format(names[i].split('.')[0])))

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
    config['data_loader']['name'], config['data_loader']['mask'], config['data_loader']['w']))

  print('using {} GPUs for testing ... '.format(ngpus_per_node))
  # setup distributed parallel training environments
  ngpus_per_node = torch.cuda.device_count()
  config['world_size'] = ngpus_per_node
  config['init_method'] = 'tcp://127.0.0.1:'+ args.port
  config['distributed'] = True
  mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
