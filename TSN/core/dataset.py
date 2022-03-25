import random
from random import shuffle
import os
import math
import numpy as np
from PIL import Image, ImageFilter
from glob import glob
import cv2

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from core.utils import ZipReader
from PIL import ImageDraw
import json

def get_coor(index, size):
  index = int(index)
  w, h = size
  return ((index % (w * h)) / h, ((index % (w * h)) % h))


def gen_input(img, xcenter=64, ycenter=64, size=45, tex_patch = None):
  w, h = 256, 256
  xstart = max(int(xcenter - size / 2), 0)
  ystart = max(int(ycenter - size / 2), 0)
  xend = min(int(xcenter + size / 2), w)
  yend = min(int(ycenter + size / 2), h)
  tex_patch[:, xstart:xend, ystart:yend] = 1
  return tex_patch


def rand_between(a, b):
  return a + torch.round(torch.rand(1) * (b - a))[0]


def gen_inputb_rand(img, seg2, num_patch=1):
  crop_size = int(rand_between(40, 60))
  c, w, h = img.size()
  tex_patch = torch.zeros(1, 256, 256)

  for i in range(1):
    for j in range(num_patch):
      # --------------------取下半身纹理------------------------#

      seg_index_size = seg2[i, :, :].view(-1).size()[0]  # 16384
      seg_index = torch.arange(0, seg_index_size)
      seg_one = seg_index[seg2[i, :, :].view(-1) == 1]

      if len(seg_one) != 0 and 1:
        # seg_select_index = int(rand_between(0, seg_one.view(-1).size()[0] - 1))
        seg_select_index = int(seg_one.view(-1).size()[0] / 2)
        x, y = get_coor(seg_one[seg_select_index], seg2[i, :, :].size())
        y = h / 2
      else:
        img = img + (-1)
        x, y = (w / 2, h / 2)

      tex_patch = gen_input(img, x, y, crop_size, tex_patch)
    # --------------------------取下半身纹理------------------------------------#
  return tex_patch

def gen_inputb_randb(img, seg2, num_patch=1):
  crop_size = int(rand_between(32, 48))
  c, w, h = img.size()
  tex_patch = torch.zeros(1, 256, 256)

  for i in range(1):
    for j in range(num_patch):
      # --------------------取下半身纹理------------------------#

      seg_index_size = seg2[i, :, :].view(-1).size()[0]  # 16384
      seg_index = torch.arange(0, seg_index_size)
      seg_one = seg_index[seg2[i, :, :].view(-1) == 1]

      if len(seg_one) != 0 and 1:
        # seg_select_index = int(rand_between(0, seg_one.view(-1).size()[0] - 1))
        seg_select_index = int(seg_one.view(-1).size()[0] / 2)
        x, y = get_coor(seg_one[seg_select_index], seg2[i, :, :].size())
        y = 100
      else:
        img = img + (-1)
        x, y = (w / 2, h / 2)

      tex_patch = gen_input(img, x, y, crop_size, tex_patch)
    # --------------------------取下半身纹理------------------------------------#
  return tex_patch


class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_args, debug=False, split='train', level=None):
    super(Dataset, self).__init__()
    self.split = split
    self.level = level
    self.w, self.h = data_args['w'], data_args['h']
    self.data = [os.path.join(data_args['zip_root'], data_args['name'], i)
      for i in np.genfromtxt(os.path.join(data_args['flist_root'], data_args['name'], split+'.flist'), dtype=np.str, encoding='utf-8')]
    self.mask_type = data_args.get('mask', 'pconv')
    if self.mask_type == 'pconv':
      self.mask = [os.path.join(data_args['zip_root'], 'mask/{}.png'.format(str(i).zfill(5))) for i in range(2000, 12000)]
      if self.level is not None:
        self.mask = [os.path.join(data_args['zip_root'], 'mask/{}.png'.format(str(i).zfill(5))) for i in range(self.level*2000, (self.level+1)*2000)]
      self.mask = self.mask*(max(1, math.ceil(len(self.data)/len(self.mask))))
    else:
      self.mask = [0]*len(self.data)
    self.data.sort()

    self.transform = transforms.Compose([transforms.ToTensor()
                                          , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    self.transform2 = transforms.Compose([transforms.ToTensor()])
    self.point_num = 18
    self.height = 256
    self.width = 256
    self.radius = 4

    if split == 'train':
      self.data = self.data*data_args['extend']
      shuffle(self.data)
    if debug:
      self.data = self.data[:100]

  def __len__(self):
    return len(self.data)

  def set_subset(self, start, end):
    self.mask = self.mask[start:end]
    self.data = self.data[start:end]

  def __getitem__(self, index):
    try:
      item = self.load_item(index)
    except:
      print('loading error: ' + self.data[index])
      item = self.load_item(0)
    return item

  def load_item(self, index):
    # load image
    img_path = os.path.dirname(self.data[index]) + '.zip'
    # print('---------------------------------------------path', img_path)
    img_name = os.path.basename(self.data[index])
    # img_name = "70881_1.jpg"
    # self.data[index] = "/data/hj/data/kt2/train/pen5/42_70880/70881_1.jpg"
    # print('---------------------------image', self.data[index])

    # img = Image.open(self.data[index]).convert('RGB')
    img2 = Image.open(self.data[index].replace('pen5', 'imageyy'))  # 读取人的图片
    img2 = self.transform(img2)

    parsing = Image.open(self.data[index].replace('pen5', 'parsing').replace('jpg', 'png'))  # 读取parsing图片
    if parsing.__class__.__name__ == 'NoneType':
      print('parsing')
    parsing = np.array(parsing).astype(np.long)
    parsing = torch.from_numpy(parsing)
    parsing_im_b1_20 = torch.zeros(20, 256, 256)
    for i in range(20):
      parsing_im_b1_20[i] += (parsing == i).float()

    # load mask
    if self.mask_type == 'pconv':
      # m_index = random.randint(0, len(self.mask)-1) if self.split == 'train' else index
      # mask_path = os.path.dirname(self.mask[m_index]) + '.zip'
      # mask_name = os.path.basename(self.mask[m_index])
      # patch_p = Image.open(self.data[index].replace('pen9', 'patch_p'))  # 读取上衣的纹理块
      # patch_p = self.transform(patch_p)
      # patch_b = Image.open(self.data[index].replace('pen9', 'patch_b'))  # 读取下装的纹理块
      # patch_b = self.transform(patch_b)
      m = np.zeros((self.h, self.w)).astype(np.uint8)
    else:
      m = np.zeros((self.h, self.w)).astype(np.uint8)
      if self.split == 'train':
        t, l = random.randint(0, self.h//2), random.randint(0, self.w//2)
        m[t:t+self.h//2, l:l+self.w//2] = 255
      else:
        m[self.h//4:self.h*3//4, self.w//4:self.w*3//4] = 255
      mask = Image.fromarray(m).convert('L')

      # load pose points
    pose_name = self.data[index].replace('pen5', 'pose').replace('.jpg', '.json')
    with open(pose_name) as f:
      pose_data = json.load(f)

    pose_maps = torch.zeros((self.point_num, self.height, self.width))
    im_pose = Image.new('RGB', (self.width, self.height))
    pose_draw = ImageDraw.Draw(im_pose)

    for i in range(self.point_num):
      one_map = Image.new('RGB', (self.width, self.height))
      draw = ImageDraw.Draw(one_map)
      if '%d' % i in pose_data:
        pointX = pose_data['%d' % i][0]
        pointY = pose_data['%d' % i][1]
        if pointX > 1 or pointY > 1:
          draw.ellipse((pointX - self.radius, pointY - self.radius, pointX + self.radius,
                        pointY + self.radius), 'white', 'white')
          pose_draw.ellipse((pointX - self.radius, pointY - self.radius, pointX + self.radius,
                             pointY + self.radius), 'white', 'white')
      one_map = self.transform(one_map)[0]
      pose_maps[i] = one_map
    im_pose_array = self.transform(im_pose)

    parsing_im = torch.zeros(1, 256, 256)
    parsing_im += (parsing == 5).float()
    parsing_im += (parsing == 6).float()
    parsing_im += (parsing == 7).float()

    parsing_imx = torch.zeros(1, 256, 256)
    parsing_imx += (parsing == 9).float()
    parsing_imx += (parsing == 12).float()

    img_t = img2 * parsing_im  # 上衣部分
    img_b = img2 * parsing_imx  # 下装部分

    mask_p = gen_inputb_rand(img_t, parsing_im.clone())
    mask_b = gen_inputb_randb(img_b, parsing_imx.clone())


    return img_name, parsing_im_b1_20, img2, parsing_im, parsing_imx, img_t * mask_p, img_b * mask_b, mask_p, mask_b

  def create_iterator(self, batch_size):
    while True:
      sample_loader = DataLoader(dataset=self,batch_size=batch_size,drop_last=True)
      for item in sample_loader:
        yield item