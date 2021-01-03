# -*- coding: utf-8 -*-

"""
    @date: 2019.07.18
    @author: samuel ko
    @func: PRNet Training Part.
"""
import os
import cv2
import random
import numpy as np
import itertools
from PIL import Image
from tqdm import tqdm

import torch
import torch.optim
from torch.autograd import Variable
from model.resfcn256 import ResFCN256
from model.discriminator import Discriminator

from tools.WLP300dataset import PRNetDataset, ToTensor, ToNormalize, ToResize
from tools.prnet_loss import WeightMaskLoss, INFO

from config.config import FLAGS

from utils.utils import save_image, test_data_preprocess, make_all_grids, make_grid
from utils.losses import SSIM

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, models
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

# Set random seem for reproducibility
manualSeed = 5
INFO("Random Seed", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def main(data_dir):
    origin_img, uv_map_gt, uv_map_predicted = None, None, None

    if not os.path.exists(FLAGS['images']):
        os.mkdir(FLAGS['images'])

    # 1) Create Dataset of 300_WLP & Dataloader.
    wlp300 = PRNetDataset(root_dir=data_dir,
                          transform=transforms.Compose([ToTensor(), ToResize((416, 416)),
                                                        ToNormalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])]))

    wlp300_dataloader = DataLoader(dataset=wlp300, batch_size=FLAGS['batch_size'], shuffle=True, num_workers=1)

    # 2) Intermediate Processing.
    transform_img = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])
    ])


    # 3) Create PRNet model.
    start_epoch, target_epoch = FLAGS['start_epoch'], FLAGS['target_epoch']
    g_x = ResFCN256(resolution_input=416, resolution_output=416, channel=3, size=16)
    g_y = ResFCN256(resolution_input=416, resolution_output=416, channel=3, size=16)
    d_x = Discriminator()
    d_y = Discriminator()

    # Load the pre-trained weight
    if FLAGS['resume'] != "" and os.path.exists(os.path.join(FLAGS['pretrained'], FLAGS['resume'])):
        state = torch.load(os.path.join(FLAGS['pretrained'], FLAGS['resume']))
        try:
            g_x.load_state_dict(state['g_x'])
            g_y.load_state_dict(state['g_y'])
            d_x.load_state_dict(state['d_x'])
            d_y.load_state_dict(state['d_y'])
        except Exception:
            g_x.load_state_dict(state['prnet'])
        start_epoch = state['start_epoch']
        INFO("Load the pre-trained weight! Start from Epoch", start_epoch)
    else:
        start_epoch = 0
        INFO("Pre-trained weight cannot load successfully, train from scratch!")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    g_x.to(FLAGS["device"])
    g_y.to(FLAGS["device"])
    d_x.to(FLAGS["device"])
    d_y.to(FLAGS["device"])

    optimizer_g = torch.optim.Adam(itertools.chain(g_x.parameters(), g_y.parameters()),
                                    lr=FLAGS["lr"], betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(itertools.chain(d_x.parameters(), d_y.parameters()),
                                    lr=FLAGS["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.99)

    stat_loss = SSIM(mask_path=FLAGS["mask_path"], gauss=FLAGS["gauss_kernel"])
    loss = WeightMaskLoss(mask_path=FLAGS["mask_path"])
    bce_loss = torch.nn.BCEWithLogitsLoss()
    bce_loss.to(FLAGS["device"])
    l1_loss = nn.L1Loss().to(FLAGS["device"])
    lambda_X = 10
    lambda_Y = 10
    #Loss function for adversarial
    for ep in range(start_epoch, target_epoch):
        bar = tqdm(wlp300_dataloader)
        loss_list_cycle_x = []
        loss_list_cycle_y = []
        loss_list_d_x = []
        loss_list_d_y = []
        real_label = torch.ones(FLAGS['batch_size'])
        fake_label = torch.zeros(FLAGS['batch_size'])
        for i, sample in enumerate(bar):
            real_y, real_x = sample['uv_map'].to(FLAGS['device']), sample['origin'].to(FLAGS['device'])
            # x -> y' -> x^
            optimizer_g.zero_grad()
            fake_y = g_x(real_x)
            prediction = d_x(fake_y)
            loss_g_x = bce_loss(prediction, real_label)
            x_hat = g_y(fake_y)
            loss_cycle_x = l1_loss(x_hat, real_x) * lambda_X
            loss_x = loss_g_x + loss_cycle_x
            loss_x.backward(retain_graph=True)
            optimizer_g.step()
            loss_list_cycle_x.append(loss_x.item())
            # y -> x' -> y^
            optimizer_g.zero_grad()
            fake_x = g_y(real_y)
            prediction = d_y(fake_x)
            loss_g_y = bce_loss(prediction, real_label)
            y_hat = g_x(fake_x)
            loss_cycle_y = l1_loss(y_hat, real_y) * lambda_Y
            loss_y = loss_g_y + loss_cycle_y
            loss_y.backward(retain_graph=True)
            optimizer_g.step()
            loss_list_cycle_y.append(loss_y.item())
            # d_x
            optimizer_d.zero_grad()
            pred_real = d_x(real_y)
            loss_d_x_real = bce_loss(pred_real, real_label)
            pred_fake = d_x(fake_y)
            loss_d_x_fake = bce_loss(pred_fake, fake_label)
            loss_d_x = (loss_d_x_real + loss_d_x_fake) * 0.5
            loss_d_x.backward()
            loss_list_d_x.append(loss_d_x.item())
            optimizer_d.step()
            if 'WGAN' in FLAGS['gan_type']:
                    for p in d_x.parameters():
                        p.data.clamp_(-1, 1)
            # d_y
            optimizer_d.zero_grad()
            pred_real = d_y(real_x)
            loss_d_y_real = bce_loss(pred_real, real_label)
            pred_fake = d_y(fake_x)
            loss_d_y_fake = bce_loss(pred_fake, fake_label)
            loss_d_y = (loss_d_y_real + loss_d_y_fake) * 0.5
            loss_d_y.backward()
            loss_list_d_y.append(loss_d_y.item())
            optimizer_d.step()
            if 'WGAN' in FLAGS['gan_type']:
                    for p in d_y.parameters():
                        p.data.clamp_(-1, 1)

        if ep % FLAGS["save_interval"] == 0:
            
            with torch.no_grad():
                print(" {} [Loss_G_X] {} [Loss_G_Y] {} [Loss_D_X] {} [Loss_D_Y] {}".format(ep, loss_list_g_x[-1], loss_list_g_y[-1], loss_list_d_x[-1], loss_list_d_y[-1]))
                origin = cv2.imread("./test_data/obama_origin.jpg")
                gt_uv_map = np.load("./test_data/test_obama.npy")
                origin, gt_uv_map = test_data_preprocess(origin), test_data_preprocess(gt_uv_map)

                origin, gt_uv_map = transform_img(origin), transform_img(gt_uv_map)

                origin_in = origin.unsqueeze_(0).cuda()
                pred_uv_map = g_x(origin_in).detach().cpu()

                save_image([origin.cpu(), gt_uv_map.unsqueeze_(0).cpu(), pred_uv_map],
                           os.path.join(FLAGS['images'], str(ep) + '.png'), nrow=1, normalize=True)

            # Save model
            print("Save model")
            state = {
                'g_x': g_x.state_dict(),
                'g_y': g_y.state_dict(),
                'd_x': d_x.state_dict(),
                'd_y': d_y.state_dict(),
                'start_epoch': ep,
            }
            torch.save(state, os.path.join(FLAGS['images'], '{}.pth'.format(ep)))

            scheduler.step()
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", help="specify input directory.")
    args = parser.parse_args()
    main(args.train_dir)
