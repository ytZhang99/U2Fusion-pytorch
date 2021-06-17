import os
import cv2
import math
import time
import torch
import random
import matplotlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms

from vgg import *
from utils import *
from option import args
from model import DenseNet
from pytorch_msssim import ssim
from dataset import MEFdataset, TestData
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

EPS = 1e-8
c = 3500


class Train(object):
    def __init__(self):
        self.num_epochs = args.epochs
        self.lr = args.lr

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        self.train_set = MEFdataset(self.transform)
        self.train_loader = data.DataLoader(self.train_set, batch_size=args.batch_size,
                                            shuffle=True, num_workers=0, pin_memory=False)

        self.model = DenseNet().cuda()
        self.feature_model = vgg16().cuda()
        self.feature_model.load_state_dict(torch.load('vgg16.pth'))
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.loss_mse = nn.MSELoss(reduction='mean').cuda()
        self.train_loss_1 = []
        self.train_loss_2 = []
        if args.validation:
            self.val_list = []
            self.best_psnr = 0

    def train(self):
        seed = args.seed
        random.seed(seed)
        torch.manual_seed(seed)
        writer = SummaryWriter(log_dir=args.log_dir, filename_suffix='train_loss')

        if os.path.exists(args.model_path + args.model):
            print('===> Loading pre-trained model......')
            state = torch.load(args.model_path + args.model)
            self.model.load_state_dict(state['model'])
            self.train_loss_1 = state['train_loss_1']
            self.train_loss_2 = state['train_loss_2']
            # self.lr = state['lr']

        for ep in range(self.num_epochs):
            ep_loss_1 = []
            ep_loss_2 = []
            for batch, (over, under) in enumerate(self.train_loader):
                over_a = (over + 1) / 2
                over_a = over_a.cuda()
                under_a = (under + 1) / 2
                under_a = under_a.cuda()

                with torch.no_grad():
                    feat_1 = torch.cat((over_a, over_a, over_a), dim=1)
                    feat_1 = self.feature_model(feat_1)
                    feat_2 = torch.cat((under_a, under_a, under_a), dim=1)
                    feat_2 = self.feature_model(feat_2)

                    for i in range(len(feat_1)):
                        m1 = torch.mean(features_grad(feat_1[i]).pow(2), dim=[1, 2, 3])
                        m2 = torch.mean(features_grad(feat_2[i]).pow(2), dim=[1, 2, 3])
                        if i == 0:
                            w1 = torch.unsqueeze(m1, dim=-1)
                            w2 = torch.unsqueeze(m2, dim=-1)
                        else:
                            w1 = torch.cat((w1, torch.unsqueeze(m1, dim=-1)), dim=-1)
                            w2 = torch.cat((w2, torch.unsqueeze(m2, dim=-1)), dim=-1)
                    weight_1 = torch.mean(w1, dim=-1) / c
                    weight_2 = torch.mean(w2, dim=-1) / c
                    weight_list = torch.cat((weight_1.unsqueeze(-1), weight_2.unsqueeze(-1)), -1)
                    weight_list = F.softmax(weight_list, dim=-1)

                over = over.cuda()
                under = under.cuda()

                self.optimizer.zero_grad()
                torch.cuda.synchronize()
                start_time = time.time()
                fused_img = self.model(over, under)
                torch.cuda.synchronize()
                end_time = time.time()
                fused_img = (fused_img + 1) / 2

                over = (over + 1) / 2
                under = (under + 1) / 2

                loss_1 = weight_list[:, 0] * (1 - ssim(fused_img, over, nonnegative_ssim=True)) \
                         + weight_list[:, 1] * (1 - ssim(fused_img, under, nonnegative_ssim=True))
                loss_1 = torch.mean(loss_1)

                loss_2 = weight_list[:, 0] * self.loss_mse(fused_img, over) \
                         + weight_list[:, 1] * self.loss_mse(fused_img, under)
                loss_2 = torch.mean(loss_2)

                loss = loss_1 + 20 * loss_2
                ep_loss_1.append(loss_1.item())
                ep_loss_2.append(loss_2.item())
                loss.backward()
                self.optimizer.step()

                if batch % 50 == 0 and batch != 0:
                    print('Epoch:{}\tcur/all:{}/{}\tLoss_1:{:.4f}\tLoss_2:{:.4f}\t'
                          'Time:{:.2f}s'.format(ep + 1, batch,
                                                len(self.train_loader),
                                                loss_1.item(),
                                                loss_2.item(),
                                                end_time - start_time))

            self.scheduler.step()
            self.train_loss_1.append(np.mean(ep_loss_1))
            self.train_loss_2.append(np.mean(ep_loss_2))

            state = {
                'model': self.model.state_dict(),
                'train_loss_1': self.train_loss_1,
                'train_loss_2': self.train_loss_2,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            torch.save(state, args.model_path + args.model)
            if ep % 5 == 0:
                torch.save(state, args.model_path + str(ep) + '.pth')
            matplotlib.use('Agg')
            fig1 = plt.figure()
            plot_loss_list_1 = self.train_loss_1
            plt.plot(plot_loss_list_1)
            plt.savefig('train_loss_curve_1.png')
            fig2 = plt.figure()
            plot_loss_list_2 = self.train_loss_2
            plt.plot(plot_loss_list_2)
            plt.savefig('train_loss_curve_2.png')

            writer.add_scalar('ssim_loss', np.mean(ep_loss_1), ep)
            writer.add_scalar('mse_loss', np.mean(ep_loss_2), ep)

            if args.train_test:
                t = Test(ep)
                t.test()
        print('===> Finished Training!')


class Test(object):
    def __init__(self, ep=None):
        self.ep = ep
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                         std=[0.5, 0.5, 0.5])])
        self.batch_size = 1
        self.test_set = TestData(self.transform)
        self.test_loader = data.DataLoader(self.test_set, batch_size=1, shuffle=False,
                                           num_workers=0, pin_memory=False)
        self.model = DenseNet().cuda()
        self.state = torch.load(args.model_path + args.model)
        self.model.load_state_dict(self.state['model'])

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for batch, imgs in enumerate(self.test_loader):
                print('Processing picture No.{}'.format(batch + 1))
                imgs = torch.squeeze(imgs, dim=0)
                img1_y = imgs[0:1, 0:1, :, :].cuda()
                img2_y = imgs[1:2, 0:1, :, :].cuda()

                img_cr = imgs[:, 1:2, :, :].cuda()
                img_cb = imgs[:, 2:3, :, :].cuda()
                w_cr = (torch.abs(img_cr) + EPS) / torch.sum(torch.abs(img_cr) + EPS, dim=0)
                w_cb = (torch.abs(img_cb) + EPS) / torch.sum(torch.abs(img_cb) + EPS, dim=0)
                fused_img_cr = torch.sum(w_cr * img_cr, dim=0, keepdim=True).clamp(-1, 1)
                fused_img_cb = torch.sum(w_cb * img_cb, dim=0, keepdim=True).clamp(-1, 1)

                fused_img_y = self.model(img1_y, img2_y)
                fused_img = torch.cat((fused_img_y, fused_img_cr, fused_img_cb), dim=1)
                fused_img = (fused_img + 1) * 127.5
                fused_img = fused_img.squeeze(0)
                fused_img = fused_img.cpu().numpy()
                fused_img = np.transpose(fused_img, (1, 2, 0))
                fused_img = fused_img.astype(np.uint8)
                fused_img = cv2.cvtColor(fused_img, cv2.COLOR_YCrCb2BGR)

                if self.ep:
                    save_path = args.save_dir + str(self.ep) + '_epoch/'
                else:
                    save_path = args.save_dir

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite((save_path + str(batch + 1) + args.ext), fused_img)
            print('Finished testing!')
