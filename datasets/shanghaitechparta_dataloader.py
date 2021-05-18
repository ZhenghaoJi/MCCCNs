import os
import random
import numpy as np
import cv2

import torch
from PIL import Image
import torchvision
import torchvision.transforms.functional as F


class CrowdDataset(torch.utils.data.Dataset):

    def __init__(self, labeled_file_list, labeled_main_transform=None, labeled_img_transform=None, labeled_dmap_transform=None):

        self.labeled_data_files = []
        with open(labeled_file_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.labeled_data_files.append(line.strip())
        f.close()

        self.label_main_transform = labeled_main_transform
        self.label_img_transform = labeled_img_transform
        self.label_dmap_transform = labeled_dmap_transform

    def __len__(self):
        return len(self.labeled_data_files)

    def __getitem__(self, index):
        index = index % len(self.labeled_data_files)
        labeled_image_filename = self.labeled_data_files[index]
        labeled_gt_filename = labeled_image_filename.replace('imgs', 'gt').replace('.jpg', '.npy')
        img = Image.open(labeled_image_filename)
        if img.mode == 'L':
            img = img.convert('RGB')
        dmap = np.load(labeled_gt_filename)
        dmap = dmap.astype(np.float32, copy=False)
        dmap = Image.fromarray(dmap)

        if self.label_main_transform is not None:
            img, dmap = self.label_main_transform((img, dmap))
        if self.label_img_transform is not None:
            img = self.label_img_transform(img)
        if self.label_dmap_transform is not None:
            dmap = self.label_dmap_transform(dmap)

        return {'image': img, 'densitymap': dmap, 'imagepath': labeled_image_filename}


def get_train_shanghaitechpartA_dataloader(labeled_file_list, use_flip, batch_size=1, mean=[0.5,0.5,0.5], std=[0.225,0.225,0.225]):
    main_transform_list = []
    if use_flip:
        main_transform_list.append(RandomHorizontalFlip())

    main_transform_list.append(PairedCrop())

    main_transform = torchvision.transforms.Compose(main_transform_list)
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    densitymap_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    dataset = CrowdDataset(
        labeled_file_list=labeled_file_list,
        labeled_main_transform=main_transform,
        labeled_img_transform=image_transform,
        labeled_dmap_transform=densitymap_transform
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader


def get_test_shanghaitechpartA_dataloader(file_list):
    main_transform_list = []
    main_transform_list.append(PairedCrop())
    main_transform = torchvision.transforms.Compose(main_transform_list)
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
    ])
    densitymap_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    dataset = CrowdDataset(
        labeled_file_list=file_list,
        labeled_main_transform=main_transform,
        labeled_img_transform=image_transform,
        labeled_dmap_transform=densitymap_transform
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    return dataloader


class RandomCrop:
    '''
    Random crop 1/2 size of the image and its corresponding density map for training
    '''

    @staticmethod
    def get_params(img, size):
        w, h = img.size
        th, tw = size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h-th+1, size=(1,)).item()
        j = torch.randint(0, w-tw+1, size=(1,)).item()

        return i,j,th,tw

    def __call__(self, img_and_dmap):

        img, dmap = img_and_dmap
        self.size = (img.size[1] // 2, img.size[0] // 2)

        # if img.size[0] < 300 or img.size[1] < 300:
        i,j,h,w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        dmap = F.crop(dmap, i, j, h, w)

        return (img, dmap)


class RandomHorizontalFlip:
    '''
    Random horizontal flip.
    probability = 0.5
    '''

    def __call__(self, data):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img, dmap = data
        if random.random() < 0.5:
            return (img.transpose(Image.FLIP_LEFT_RIGHT), dmap.transpose(Image.FLIP_LEFT_RIGHT))
        else:
            return (img, dmap)


class PairedCrop:
    '''
    Paired Crop for both image and its density map.
    Note that due to the maxpooling in the nerual network,
    we must promise that the size of input image is the corresponding factor.
    '''

    def __init__(self, factor=8): # since the CSRNet uses Maxpooling three times in the frontend layers.
        self.factor = factor

    @staticmethod
    def get_params(img, factor):
        w, h = img.size
        if w % factor == 0 and h % factor == 0:
            return 0, 0, h, w
        else:
            return 0, 0, h - (h % factor), w - (w % factor)

    def __call__(self, data):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img, dmap = data

        i, j, th, tw = self.get_params(img, self.factor)

        img = F.crop(img, i, j, th, tw)
        dmap = F.crop(dmap, i, j, th, tw)
        return (img, dmap)


