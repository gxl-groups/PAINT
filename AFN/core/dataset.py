import random
from random import shuffle
import os
import math
import numpy as np
from PIL import Image, ImageFilter

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def rand_between(a, b):
  return a + torch.round(torch.rand(1) * (b - a))[0]

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
    # print('---------------------------------------------path', img_path)
    img_name = os.path.basename(self.data[index])
    # img = Image.open(self.data[index]).convert('RGB')
    img_split = img_name.split('_')
    id = int((img_split[0])) + 2
    id1 = int((img_split[0])) + 1

    image_n = str(id) + '_3.jpg'
    image_n1 = str(id1) + '_2.jpg'
    image_name0 = self.data[index]
    image_name = self.data[index].replace(img_name, image_n)
    image_name1 = self.data[index].replace(img_name, image_n1)

    ir_id = int(rand_between(0, 2))
    # ir_id = 1
    if (ir_id == 1):
      temp = image_name0
      image_name0 = image_name1
      image_name1 = temp
    if (ir_id == 2):
      temp = image_name0
      image_name0 = image_name
      image_name = temp

    ir_id = int(rand_between(0, 2))
    if (ir_id == 1):
      temp = image_name1
      image_name1 = image_name
      image_name = temp

    parsing = Image.open(image_name0.replace('pen55', 'parsing').replace('jpg', 'png'))  # 读取parsing图片
    if parsing.__class__.__name__ == 'NoneType':
      print('parsing')
    parsing = np.array(parsing).astype(np.long)
    parsing = torch.from_numpy(parsing)
    parsing_im_b_20 = torch.zeros(9, 256, 256)
    parsing_im_b_20[0] += (parsing == 2).float()
    parsing_im_b_20[1] += (parsing == 14).float() + (parsing == 15).float() +(parsing == 10).float()
    parsing_im_b_20[2] += (parsing == 16).float() + (parsing == 17).float() + (parsing == 8).float()
    parsing_im_b_20[3] += (parsing == 1).float()
    parsing_im_b_20[4] += (parsing == 5).float() + (parsing == 7).float()
    parsing_im_b_20[5] += (parsing == 6).float() + (parsing == 9).float() +(parsing == 12).float()
    parsing_im_b_20[6] += (parsing == 18).float() + (parsing == 19).float()
    parsing_im_b_20[7] += (parsing == 13).float() + (parsing == 3).float()
    parsing_im_b_20[8] += (parsing == 0).float() + (parsing == 4).float() + \
                          (parsing == 11).float()

    parsingp1 = Image.open(image_name1.replace('pen55', 'parsing').replace('jpg', 'png'))  # 读取pose2的parsing图片
    parsingp1 = np.array(parsingp1).astype(np.long)
    parsingp1 = torch.from_numpy(parsingp1)
    parsing_im_bp1_20 = torch.zeros(9, 256, 256)
    parsing_im_bp1_20[0] += (parsingp1 == 2).float()
    parsing_im_bp1_20[1] += (parsingp1 == 14).float() + (parsingp1 == 15).float() + (parsingp1 == 10).float()
    parsing_im_bp1_20[2] += (parsingp1 == 16).float() + (parsingp1 == 17).float() + (parsingp1 == 8).float()
    parsing_im_bp1_20[3] += (parsingp1 == 1).float()
    parsing_im_bp1_20[4] += (parsingp1 == 5).float() + (parsingp1 == 7).float()
    parsing_im_bp1_20[5] += (parsingp1 == 6).float() + (parsingp1 == 9).float() + (parsingp1 == 12).float()
    parsing_im_bp1_20[6] += (parsingp1 == 18).float() + (parsingp1 == 19).float()
    parsing_im_bp1_20[7] += (parsingp1 == 13).float() + (parsingp1 == 3).float()
    parsing_im_bp1_20[8] += (parsingp1 == 0).float() + (parsingp1 == 4).float() + \
                            (parsingp1 == 11).float()

    parsingp2 = Image.open(image_name.replace('pen55', 'parsing').replace('jpg', 'png'))  # 读取pose3的parsing图片
    parsingp2 = np.array(parsingp2).astype(np.long)
    parsingp2 = torch.from_numpy(parsingp2)
    parsing_im_bp2_20 = torch.zeros(9, 256, 256)
    parsing_im_bp2_20[0] += (parsingp2 == 2).float()
    parsing_im_bp2_20[1] += (parsingp2 == 14).float() + (parsingp2 == 15).float() + (parsingp2 == 10).float()
    parsing_im_bp2_20[2] += (parsingp2 == 16).float() + (parsingp2 == 17).float() + (parsingp2 == 8).float()
    parsing_im_bp2_20[3] += (parsingp2 == 1).float()
    parsing_im_bp2_20[4] += (parsingp2 == 5).float() + (parsingp2 == 7).float()
    parsing_im_bp2_20[5] += (parsingp2 == 6).float() + (parsingp2 == 9).float() + (parsingp2 == 12).float()
    parsing_im_bp2_20[6] += (parsingp2 == 18).float() + (parsingp2 == 19).float()
    parsing_im_bp2_20[7] += (parsingp2 == 13).float() + (parsingp2 == 3).float()
    parsing_im_bp2_20[8] += (parsingp2 == 0).float() + (parsingp2 == 4).float() + \
                            (parsingp2 == 11).float()

    img2 = Image.open(image_name0.replace('pen55', 'imageyy')).convert('RGB')  # 读取人的图片
    img2 = self.transform(img2)
    imgp1 = Image.open(image_name1.replace('pen55', 'imageyy')).convert('RGB')  # 读取人pose2的图片
    imgp1 = self.transform(imgp1)
    imgp2 = Image.open(image_name.replace('pen55', 'imageyy')).convert('RGB')  # 读取人pose3的图片
    imgp2 = self.transform(imgp2)


    return img_name, parsing_im_b_20, img2, imgp1, parsing_im_bp1_20, imgp2, parsing_im_bp2_20

  def create_iterator(self, batch_size):
    while True:
      sample_loader = DataLoader(dataset=self,batch_size=batch_size,drop_last=True)
      for item in sample_loader:
        yield item