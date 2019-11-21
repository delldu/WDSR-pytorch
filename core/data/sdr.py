#! /usr/bin/env python
# coding=utf-8
#
# /************************************************************************************
# ***
# ***   File Author: Dell, Sat Nov  2 15:26:20 CST 2019
# ***
# ************************************************************************************/
#

"""
    Access SDR data as CxHxW tensors.
"""

import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import random
import pdb

DATA_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
LP_ROOTDIR = "/home/dell/ZDisk/WorkSpace/4K/dataset/SDR_540p"
LC_ROOTDIR = "/home/dell/ZDisk/WorkSpace/4K/dataset/SDR_540c"
LX_ROOTDIR = "/home/dell/ZDisk/WorkSpace/4K/dataset/SDR_540x"

HR_ROOTDIR = "/home/dell/ZDisk/WorkSpace/4K/dataset/SDR_4K"

TRAIN_FILE_LIST = "SDR_train_list.txt"
VALID_FILE_LIST = "SDR_valid_list.txt"
TEST_FILE_LIST = "SDR_test_list.txt"

# Big patch size
PATCH_SIZE = 96
PATCH_SCALE = 4
CACHE_SIZE = 32

torch.manual_seed(42)


def dn_image_p(img):
    """Transform LP to LC image."""
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img.filter(ImageFilter.GaussianBlur(radius=1))


def dn_tensor_p(tensor):
    """Transform LP to LC tensor, CxHxW, 0.0~1.0."""
    image = transforms.ToPILImage()(tensor)
    image = dn_image_p(image)
    return transforms.ToTensor()(image)


def sr_image_p(img):
    """Transform LR to HR image."""
    oh, ow = img.height * PATCH_SCALE, img.width * PATCH_SCALE
    return img.resize((ow, oh), resample=Image.BICUBIC)


def sr_tensor_p(tensor):
    """Transform LR to HR tensor."""
    image = transforms.ToPILImage()(tensor)
    image = sr_image_p(image)
    return transforms.ToTensor()(image)


def aug_patches(patches, hflip=True, vflip=True, rot=True):
    """Augment patches, here patches is a list of PIL images."""
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    ret = []
    for p in patches:
        if hflip:
            p = p.transpose(Image.FLIP_LEFT_RIGHT)
        if vflip:
            p = p.transpose(Image.FLIP_TOP_BOTTOM)
        if rot90:
            p = p.transpose(Image.ROTATE_90)
        ret.append(p)
    return ret


def get_patch(lr, hr, patch_size, scale, augment=False):
    """Get random patch.

    # lr, hr are PIL Images: CxHxW, scale == 4 or 1 !
    """
    lr_h, lr_w = lr.height, lr.width
    lr_p = patch_size // scale
    lr_x = random.randrange(0, lr_w - lr_p + 1)
    lr_y = random.randrange(0, lr_h - lr_p + 1)

    hr_x, hr_y, hr_p = lr_x * scale, lr_y * scale, patch_size

    lr = lr.crop((lr_x, lr_y, lr_x + lr_p, lr_y + lr_p))
    hr = hr.crop((hr_x, hr_y, hr_x + hr_p, hr_y + hr_p))

    if augment:
        lr, hr = aug_patches([lr, hr])

    return lr, hr


def get_filelist(filename):
    """File list like 10091373:008,010,022,031,044,053,064,079,080,090."""
    data = []
    with open(filename, 'r') as f:
        filelist = [x.strip() for x in f.readlines()]
    for fields in filelist:
        vid = fields.split(':')[0]
        frame_list = fields.split(':')[1].split(',')
        for f in frame_list:
            data.append(vid + "/" + f + ".png")
    return data


class ACDataset(data.Dataset):
    """
    Auto Color Dataset.

    Data source: lg ---> lc
    gray --> color

    DESTION_SDR_540c_DIR=${HOME}/ZDisk/WorkSpace/4K/dataset/SDR_540c

    """

    def __init__(self, phase, lc_rootdir=LC_ROOTDIR):
        """Class init."""
        super(ACDataset, self).__init__()

        self.phase = phase
        self.lc_rootdir = lc_rootdir

        if self.phase == "train":
            self.keyf_filelist = os.path.join(DATA_ROOT_DIR, TRAIN_FILE_LIST)
        elif self.phase == "valid":
            self.keyf_filelist = os.path.join(DATA_ROOT_DIR, VALID_FILE_LIST)
        else:
            self.keyf_filelist = os.path.join(DATA_ROOT_DIR, TEST_FILE_LIST)
        self.samples = get_filelist(self.keyf_filelist)

        self.image2tensor = transforms.ToTensor()
        self.tensor2image = transforms.ToPILImage()

    def __getitem__(self, index):
        """Get two tensors: CxHxW, data range[0.0, 1.0]."""
        index = index % len(self.samples)
        lc = Image.open(os.path.join(self.lc_rootdir,
                                     self.samples[index])).convert("RGB")
        if self.phase == "train" or self.phase == "valid":
            lg, lc = get_patch(lc, lc, PATCH_SIZE, 1, augment=False)
        else:
            lg = lc
        lg = lg.convert('L')
        lg, lc = self.image2tensor(lg), self.image2tensor(lc)
        # lg, lc format: Tensor, [0.0, 1.0], CxHxW format
        return lg, lc

    def __len__(self):
        """Get length of dataset."""
        return len(self.samples)

    def show(self, index):
        """Show image."""
        lg, lc = self.__getitem__(index)
        # CxHxW format, [0.0, 1.0]
        image = self.tensor2image(lg)
        image.show()
        image = self.tensor2image(lc)
        image.show()


class DNDataset(data.Dataset):
    """
    Denoise Dataset.

    Data source: lp(540p) --> lc(540c), sample with big patches.

    DESTION_SDR_540p_DIR=${HOME}/ZDisk/WorkSpace/4K/dataset/SDR_540p
    DESTION_SDR_540c_DIR=${HOME}/ZDisk/WorkSpace/4K/dataset/SDR_540c

    """

    def __init__(self, phase, lp_rootdir=LP_ROOTDIR, lc_rootdir=LC_ROOTDIR):
        """Class init."""
        super(DNDataset, self).__init__()

        self.phase = phase
        self.lp_rootdir = lp_rootdir
        self.lc_rootdir = lc_rootdir

        if self.phase == "train":
            self.keyf_filelist = os.path.join(DATA_ROOT_DIR, TRAIN_FILE_LIST)
        elif self.phase == "valid":
            self.keyf_filelist = os.path.join(DATA_ROOT_DIR, VALID_FILE_LIST)
        else:
            self.keyf_filelist = os.path.join(DATA_ROOT_DIR, TEST_FILE_LIST)
        self.samples = get_filelist(self.keyf_filelist)

        self.image2tensor = transforms.ToTensor()
        self.tensor2image = transforms.ToPILImage()

    def __getitem__(self, index):
        """Get two tensors: CxHxW, data range[0.0, 1.0]."""
        index = index % len(self.samples)
        lp = Image.open(os.path.join(self.lp_rootdir,
                                     self.samples[index])).convert("RGB")
        lc = Image.open(os.path.join(self.lc_rootdir,
                                     self.samples[index])).convert("RGB")
        if self.phase == "train" or self.phase == "valid":
            lp, lc = get_patch(lp, lc, PATCH_SIZE, 1, augment=False)
        px = dn_image_p(lp)

        lp, lc, px = self.image2tensor(lp), self.image2tensor(lc), self.image2tensor(px)
        # lp, lc format: Tensor, [0.0, 1.0], CxHxW format
        return lp, lc, px

    def __len__(self):
        """Get length of dataset."""
        return len(self.samples)

    def show(self, index):
        """Show image."""
        lr, hr, px = self.__getitem__(index)
        # CxHxW format, [0.0, 1.0]
        image = self.tensor2image(lr)
        image.show()
        image = self.tensor2image(hr)
        image.show()
        image = self.tensor2image(px)
        image.show()


class SRDataset(data.Dataset):
    """Super Resolution Dataset.

    Data source: lc(540c) --> hr(4K), end to end model

    LC_ROOTDIR=${HOME}/ZDisk/WorkSpace/4K/dataset/SDR_540c
    HR_ROOTDIR=${HOME}/ZDisk/WorkSpace/4K/dataset/SDR_4K
    """

    # xxxx9999 LX_ROOTDIR
    def __init__(self, phase, lx_rootdir=LX_ROOTDIR, hr_rootdir=HR_ROOTDIR):
        """Class init."""
        super(SRDataset, self).__init__()

        self.phase = phase
        self.lx_rootdir = lx_rootdir
        self.hr_rootdir = hr_rootdir

        if self.phase == "train":
            self.keyf_filelist = os.path.join(DATA_ROOT_DIR, TRAIN_FILE_LIST)
        elif self.phase == "valid":
            self.keyf_filelist = os.path.join(DATA_ROOT_DIR, VALID_FILE_LIST)
        else:
            self.keyf_filelist = os.path.join(DATA_ROOT_DIR, TEST_FILE_LIST)

        self.samples = []
        all_files = get_filelist(self.keyf_filelist)
        for f in all_files:
            f1 = os.path.join(self.lx_rootdir, f)
            f2 = os.path.join(self.hr_rootdir, f)
            if os.path.exists(f1) and os.path.exists(f2):
                self.samples.append(f)
        # self.samples = get_filelist(self.keyf_filelist)

        self.image2tensor = transforms.ToTensor()
        self.tensor2image = transforms.ToPILImage()

        self.lr_cache = torch.FloatTensor(CACHE_SIZE, 3, PATCH_SIZE//PATCH_SCALE, PATCH_SIZE//PATCH_SCALE)
        self.hr_cache = torch.FloatTensor(CACHE_SIZE, 3, PATCH_SIZE, PATCH_SIZE)
        self.cache_index = CACHE_SIZE

    def create_cache_data(self):
        """Cache, index is file index."""
        index = int(random.random() * len(self.samples))
        index = index % len(self.samples)
        lr_image = Image.open(os.path.join(self.lx_rootdir, self.samples[index])).convert("RGB")
        hr_image = Image.open(os.path.join(self.hr_rootdir, self.samples[index])).convert("RGB")
        for i in range(CACHE_SIZE):
            lr_patch, hr_patch = get_patch(lr_image, hr_image, PATCH_SIZE, PATCH_SCALE, augment=True)
            self.lr_cache[i] = self.image2tensor(lr_patch)
            self.hr_cache[i] = self.image2tensor(hr_patch)
        # lr, hr format: Tensor, [0.0, 1.0], CxHxW format
        self.cache_index = 0

    def __getitem__(self, index):
        """Get two tensors: CxHxW, data range[0.0, 1.0]."""
        if self.phase == "test":
            lr_image = Image.open(os.path.join(self.lx_rootdir,
                                         self.samples[index])).convert("RGB")
            hr_image = Image.open(os.path.join(self.hr_rootdir,
                                         self.samples[index])).convert("RGB")

            lr = self.image2tensor(lr_image)
            hr = self.image2tensor(hr_image)
            return lr.mul(255.0), hr.mul(255.0)

        # else ... self.phase == "train" or "valid", cache empty ?
        if self.cache_index >= CACHE_SIZE:
            self.create_cache_data()

        lr = self.lr_cache[self.cache_index]
        hr = self.hr_cache[self.cache_index]
        self.cache_index = self.cache_index + 1

        return lr.mul(255.0), hr.mul(255.0)

    def __len__(self):
        """Get length of dataset."""
        return len(self.samples)

    def show(self, index):
        """Show image."""
        lr, hr = self.__getitem__(index)
        # CxHxW format, [0.0, 1.0]
        image = self.tensor2image(lr)
        image.show()
        image = self.tensor2image(hr)
        image.show()


def test_acset():
    """Test Auto Color Dataset."""
    dataset = ACDataset(phase="test")
    print(len(dataset))
    dataset.show(10)


def test_dnset():
    """Test Denoise Dataset."""
    dataset = DNDataset(phase="test")
    print(len(dataset))
    dataset.show(10)


def test_srset():
    """Test Super Resolution Dataset."""
    dataset = SRDataset(phase="train")
    print(len(dataset))
    dataset.show(10)


if __name__ == '__main__':
    # test_acset()
    # test_dnset()
    test_srset()
