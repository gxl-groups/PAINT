import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
import torch
import torchvision.transforms as transforms
import os
from PIL import ImageDraw
import numpy as np
import json
from PIL import Image
import cv2

def rand_between(a, b):
  return a + torch.round(torch.rand(1) * (b - a))[0]

class AlignedDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.data_root = opt.dataroot
        self.mode = opt.phase # train or test

        self.transform = transforms.Compose([transforms.ToTensor()
                            ,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.transform2 = transforms.Compose([transforms.ToTensor()])
        self.point_num = opt.point_num
        self.height = 256
        self.width = 256
        self.data_list = []
        self.radius = 4

        # get data_list
        with open(os.path.join(self.data_root, 'all_train.txt'), 'r') as f:
            for line in f.readlines():
                c = line.strip().split()
                if 'train' == self.mode:
                    self.data_list.append(c[0])
                else:
                    self.data_list.append(c[0])

        print(len(self.data_list))

    def __getitem__(self, index):
        item_list = self.data_list[index]

        img_split = item_list.split('_')

        ir_id = int(rand_between(0, 2))
        if (ir_id == 0):
            id = int((img_split[1].split('/'))[0]) + 1
            image_name = img_split[0] + '_' + (img_split[1].split('/'))[0] + '/' + str(id) + '_1.jpg'
        if (ir_id == 1):
            id = int((img_split[1].split('/'))[0]) + 2
            image_name = img_split[0] + '_' + (img_split[1].split('/'))[0] + '/' + str(id) + '_2.jpg'
        if (ir_id == 2):
            id = int((img_split[1].split('/'))[0]) + 3
            image_name = img_split[0] + '_' + (img_split[1].split('/'))[0] + '/' + str(id) + '_3.jpg'

        im_A_name = image_name.replace('.jpg', '.png')  # pose_A

        parsing = Image.open(os.path.join(self.data_root, 'parsing', im_A_name))
        if parsing.__class__.__name__ == 'NoneType':
            print(self.data_root, self.mode, 'parsing', im_A_name)
        parsing = np.array(parsing).astype(np.long)
        parsing = torch.from_numpy(parsing)
        parsing_im_b_20 = torch.zeros(20, 256, 256)
        for i in range(20):
            parsing_im_b_20[i] += (parsing == i).float()

        sketch = Image.open(os.path.join(self.data_root, 'canny_train', image_name))
        sketch = self.transform2(sketch)

        # 上装
        parsing_im_b_201 = torch.zeros(3, 256, 256)
        for i in range(0, 3):
            parsing_im_b_201[i] += (parsing == i + 5).float()
        parsing_im_b = torch.zeros(256, 256)
        for i in range(3):
            parsing_im_b = parsing_im_b + parsing_im_b_201[i]
        parsing_im_b = parsing_im_b.numpy()
        parsing_im_b = parsing_im_b.astype("uint8")
        sketch201 = cv2.Canny(parsing_im_b, 0, 1)
        sketch201 = torch.from_numpy(sketch201)
        sketch201 = sketch201.unsqueeze(dim=0).float() / 255
        parsing_im_b_201 = torch.cat((parsing_im_b_201, sketch201), 0)

        # 下装
        parsing_im_b_202 = torch.zeros(3, 256, 256)
        parsing_im_b_202[0] += (parsing == 8).float()
        parsing_im_b_202[1] += (parsing == 9).float()
        parsing_im_b_202[2] += (parsing == 12).float()
        parsing_im_b = torch.zeros(256, 256)
        for i in range(3):
            parsing_im_b = parsing_im_b + parsing_im_b_202[i]
        parsing_im_b = parsing_im_b.numpy()
        parsing_im_b = parsing_im_b.astype("uint8")
        sketch202 = cv2.Canny(parsing_im_b, 0, 1)
        sketch202 = torch.from_numpy(sketch202)
        # parsing_im_b = torch.zeros(1, 256, 256)
        # for i in range(3):
        #     parsing_im_b = parsing_im_b + parsing_im_b_202[i]
        # sketch202 = sketch * parsing_im_b
        sketch202 = sketch202.unsqueeze(dim=0).float() / 255

        parsing_im_b_202 = torch.cat((parsing_im_b_202, sketch202), 0)

        # 头部
        parsing_im_b_203 = torch.zeros(3, 256, 256)
        parsing_im_b_203[0] += (parsing == 1).float()
        parsing_im_b_203[1] += (parsing == 2).float()
        parsing_im_b_203[2] += (parsing == 13).float()
        parsing_im_b = torch.zeros(256, 256)
        for i in range(3):
            parsing_im_b = parsing_im_b + parsing_im_b_203[i]
        parsing_im_b = parsing_im_b.numpy()
        parsing_im_b = parsing_im_b.astype("uint8")
        sketch203 = cv2.Canny(parsing_im_b, 0, 1)
        sketch203 = torch.from_numpy(sketch203)
        # parsing_im_b = torch.zeros(1, 256, 256)
        # for i in range(3):
        #     parsing_im_b = parsing_im_b + parsing_im_b_203[i]
        # sketch203 = sketch * parsing_im_b
        sketch203 = sketch203.unsqueeze(dim=0).float() / 255
        parsing_im_b_203 = torch.cat((parsing_im_b_203, sketch203), 0)

        # 上肢
        parsing_im_b_204 = torch.zeros(3, 256, 256)
        parsing_im_b_204[0] += (parsing == 10).float()
        parsing_im_b_204[1] += (parsing == 14).float()
        parsing_im_b_204[2] += (parsing == 15).float()
        # parsing_im_b = torch.zeros(1, 256, 256)
        # for i in range(3):
        #     parsing_im_b = parsing_im_b + parsing_im_b_204[i]
        # sketch204 = sketch * parsing_im_b
        # parsing_im_b_204 = torch.cat((parsing_im_b_204, sketch204), 0)

        # 下肢
        parsing_im_b_205 = torch.zeros(3, 256, 256)
        parsing_im_b_205[0] += (parsing == 11).float()
        parsing_im_b_205[1] += (parsing == 16).float()
        parsing_im_b_205[2] += (parsing == 17).float()
        parsing_im_b = torch.zeros(1, 256, 256)
        for i in range(3):
            parsing_im_b = parsing_im_b + parsing_im_b_205[i]
        sketch205 = sketch * parsing_im_b
        parsing_im_b_205 = torch.cat((parsing_im_b_205, sketch205), 0)


        # load pose points
        pose_name = im_A_name.replace('.png', '.json')
        with open(os.path.join(self.data_root, 'pose', pose_name)) as f:
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

        return {'A': pose_maps, 'B': parsing_im_b_20, 'A_vis': im_pose_array, 'B_vis': parsing_im_b_20, 'ct': parsing_im_b_201, 'cb':parsing_im_b_202, 'c3': parsing_im_b_203, 'c4':parsing_im_b_204, 'c5': parsing_im_b_205, 'sketch':sketch}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data_list)
