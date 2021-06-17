import os
import cv2
import torch
import random
import torch.utils.data as data

from option import args


class MEFdataset(data.Dataset):
    def __init__(self, transform):
        super(MEFdataset, self).__init__()
        self.dir_prefix = args.dir_train
        self.over = os.listdir(self.dir_prefix + 'over/')
        self.under = os.listdir(self.dir_prefix + 'under/')

        self.patch_size = args.patch_size
        self.transform = transform

    def __len__(self):
        assert len(self.over) == len(self.under)
        return len(self.over)

    def __getitem__(self, idx):
        over = cv2.imread(self.dir_prefix + 'over/' + self.over[idx])
        over = cv2.cvtColor(over, cv2.COLOR_BGR2YCrCb)
        over = over[:, :, 0:1]
        under = cv2.imread(self.dir_prefix + 'under/' + self.under[idx])
        under = cv2.cvtColor(under, cv2.COLOR_BGR2YCrCb)
        under = under[:, :, 0:1]

        over_p, under_p = self.get_patch(over, under)
        if self.transform:
            over_p = self.transform(over_p)
            under_p = self.transform(under_p)

        return over_p, under_p

    def get_patch(self, over, under):
        h, w = over.shape[:2]
        stride = self.patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        over = over[y:y + stride, x:x + stride, :]
        under = under[y:y + stride, x:x + stride, :]

        return over, under


class TestData(data.Dataset):
    def __init__(self, transform):
        super(TestData, self).__init__()
        self.transform = transform
        self.dir_prefix = args.dir_test
        self.over_dir = os.listdir(self.dir_prefix + 'over/')
        self.under_dir = os.listdir(self.dir_prefix + 'under/')

    def __getitem__(self, idx):
        over = cv2.imread(self.dir_prefix + 'over/' + self.over_dir[idx])
        under = cv2.imread(self.dir_prefix + 'under/' + self.under_dir[idx])
        over_img = cv2.cvtColor(over, cv2.COLOR_BGR2YCrCb)
        under_img = cv2.cvtColor(under, cv2.COLOR_BGR2YCrCb)

        if self.transform:
            over_img = self.transform(over_img)
            under_img = self.transform(under_img)

        img_stack = torch.stack((over_img, under_img), 0)
        return img_stack

    def __len__(self):
        assert len(self.over_dir) == len(self.under_dir)
        return len(self.over_dir)
