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
    # 0) Tensoboard Writer.
    writer = SummaryWriter(FLAGS['summary_path'])
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
        transforms.Normalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])
    ])

    # 3) Create PRNet model.
    start_epoch, target_epoch = FLAGS['start_epoch'], FLAGS['target_epoch']
    model = ResFCN256(resolution_input=416, resolution_output=416, channel=3, size=16)
    discriminator = Discriminator()

    # Load the pre-trained weight
    if FLAGS['resume'] != "" and os.path.exists(os.path.join(FLAGS['pretrained'], FLAGS['resume'])):
        state = torch.load(os.path.join(FLAGS['pretrained'], FLAGS['resume']))
        model.load_state_dict(state['prnet'])
        start_epoch = state['start_epoch']
        INFO("Load the pre-trained weight! Start from Epoch", start_epoch)
    else:
        start_epoch = 0
        INFO("Pre-trained weight cannot load successfully, train from scratch!")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(FLAGS["device"])
    discriminator.to(FLAGS["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS["lr"], betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=FLAGS["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    stat_loss = SSIM(mask_path=FLAGS["mask_path"], gauss=FLAGS["gauss_kernel"])
    loss = WeightMaskLoss(mask_path=FLAGS["mask_path"])
    bce_loss = torch.nn.BCEWithLogitsLoss()
    bce_loss.to(FLAGS["device"])
    
    #Loss function for adversarial
    for ep in range(start_epoch, target_epoch):
        bar = tqdm(wlp300_dataloader)
        loss_list_G, stat_list = [], []
        loss_list_D = []
        for i, sample in enumerate(bar):
            uv_map, origin = sample['uv_map'].to(FLAGS['device']), sample['origin'].to(FLAGS['device'])

            # Inference.
            optimizer.zero_grad()
            uv_map_result = model(origin)

            # Update D
            optimizer_D.zero_grad()
            fake_detach = uv_map_result.detach()
            d_fake = discriminator(fake_detach)
            d_real = discriminator(uv_map)
            retain_graph = False
            if FLAGS['gan_type'] == 'GAN':
                loss_d = bce_loss(d_real, d_fake)
            elif FLAGS['gan_type'].find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if FLAGS['gan_type'].find('GP') >= 0:
                    epsilon = torch.rand(fake_detach.shape[0]).view(-1, 1, 1, 1)
                    epsilon = epsilon.to(fake_detach.device)
                    hat = fake_detach.mul(1 - epsilon) + uv_map.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty
            # from ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
            elif FLAGS['gan_type'] == 'RGAN':
                better_real = d_real - d_fake.mean(dim=0, keepdim=True)
                better_fake = d_fake - d_real.mean(dim=0, keepdim=True)
                loss_d = bce_loss(better_real, better_fake)
                retain_graph = True
            

            if discriminator.training:
                loss_list_D.append(loss_d.item())
                loss_d.backward(retain_graph=retain_graph)
                optimizer_D.step()

                if 'WGAN' in FLAGS['gan_type']:
                    for p in discriminator.parameters():
                        p.data.clamp_(-1, 1)
            
            # Update G
            d_fake_bp = discriminator(uv_map_result)      # for backpropagation, use fake as it is
            if FLAGS['gan_type'] == 'GAN':
                label_real = torch.ones_like(d_fake_bp)
                loss_g = bce_loss(d_fake_bp, label_real)
            elif FLAGS['gan_type'].find('WGAN') >= 0:
                loss_g = -d_fake_bp.mean()
            elif FLAGS['gan_type'] == 'RGAN':
                better_real = d_real - d_fake_bp.mean(dim=0, keepdim=True)
                better_fake = d_fake_bp - d_real.mean(dim=0, keepdim=True)
                loss_g = bce_loss(better_fake, better_real)
            
            loss_g.backward()
            loss_list_G.append(loss_g.item())
            optimizer.step()
 
            
            stat_logit = stat_loss(uv_map_result, uv_map)
            stat_list.append(stat_logit.item())
            #bar.set_description(" {} [Loss(Paper)] {} [Loss(D)] {} [SSIM({})] {}".format(ep, loss_list_G[-1], loss_list_D[-1],FLAGS["gauss_kernel"], stat_list[-1]))
            # Record Training information in Tensorboard.
            """
            if origin_img is None and uv_map_gt is None:
                origin_img, uv_map_gt = origin, uv_map
            uv_map_predicted = uv_map_result

            writer.add_scalar("Original Loss", loss_list_G[-1], FLAGS["summary_step"])
            writer.add_scalar("D Loss", loss_list_D[-1], FLAGS["summary_step"])
            writer.add_scalar("SSIM Loss", stat_list[-1], FLAGS["summary_step"])

            grid_1, grid_2, grid_3 = make_grid(origin_img, normalize=True), make_grid(uv_map_gt), make_grid(uv_map_predicted)

            writer.add_image('original', grid_1, FLAGS["summary_step"])
            writer.add_image('gt_uv_map', grid_2, FLAGS["summary_step"])
            writer.add_image('predicted_uv_map', grid_3, FLAGS["summary_step"])
            writer.add_graph(model, uv_map)
            """

        if ep % FLAGS["save_interval"] == 0:
            
            with torch.no_grad():
                print(" {} [Loss(Paper)] {} [Loss(D)] {} [SSIM({})] {}".format(ep, loss_list_G[-1], loss_list_D[-1],FLAGS["gauss_kernel"], stat_list[-1]))
                origin = cv2.imread("./test_data/obama_origin.jpg")
                gt_uv_map = np.load("./test_data/test_obama.npy")
                origin, gt_uv_map = test_data_preprocess(origin), test_data_preprocess(gt_uv_map)

                origin, gt_uv_map = transform_img(origin), transform_img(gt_uv_map)

                origin_in = origin.unsqueeze_(0).cuda()
                pred_uv_map = model(origin_in).detach().cpu()

                save_image([origin.cpu(), gt_uv_map.unsqueeze_(0).cpu(), pred_uv_map],
                           os.path.join(FLAGS['images'], str(ep) + '.png'), nrow=1, normalize=True)

            # Save model
            print("Save model")
            state = {
                'prnet': model.state_dict(),
                'Loss': loss_list_G,
                'start_epoch': ep,
                'Loss_D':loss_list_D,
            }
            torch.save(state, os.path.join(FLAGS['images'], '{}.pth'.format(ep)))

            scheduler.step()
        
    writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", help="specify input directory.")
    args = parser.parse_args()
    main(args.train_dir)
