import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
import torch
import torchvision.transforms as transforms
import os
from PIL import ImageDraw
import numpy as np
import json
from PIL import Image


class AlignedDataset(BaseDataset):

    def __init__(self, opt):
        super(AlignedDataset, self).__init__()
        self.data_root = opt.data_root
        self.mode = opt.mode # train or test

        self.transform = transforms.Compose([transforms.ToTensor()
                            ,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
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
        id = int((img_split[1].split('/'))[0]) + 1
        image_name = img_split[0] + '_' + (img_split[1].split('/'))[0] + '/' + str(id) + '_1.jpg'

        im_A_name = image_name.replace('.jpg', '.png')  # pose_A
        print('------------imaname---------------', im_A_name)

        im_A_parsing = Image.open(os.path.join(self.data_root, 'parsing', im_A_name))

        # im_B_parsing = Image.open(os.path.join(self.data_root,self.mode,'parsing',im_B_name))

        if im_A_parsing.__class__.__name__ == 'NoneType':
            print(self.data_root, self.mode, 'parsing', im_A_name)
        im_A_parsing = np.array(im_A_parsing).astype(np.long)
        im_A_parsing = torch.from_numpy(im_A_parsing)

        parsing_im_b_20 = torch.zeros((20, self.height, self.width))
        for i in range(20):
            parsing_im_b_20[i] += (im_A_parsing[i] == i).float() * i

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

        return {'A': pose_maps, 'B': parsing_im_b_20, 'A_vis': im_pose_array, 'B_paths': im_A_parsing}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data_list)