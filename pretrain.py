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
from model.unet import UNet
from model.resfcn256 import ResFCN256

from tools.WLP300dataset import PRNetDataset, ToTensor, ToNormalize, ToResize
from tools.prnet_loss import WeightMaskLoss, INFO

from config.config import FLAGS

from utils.utils import save_image, test_data_preprocess, make_all_grids, make_grid
from utils.losses import SSIM

from torchvision import transforms
from torch.utils.data import DataLoader

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
                          transform=transforms.Compose([ToTensor(), ToResize((256, 256)),
                                                        ToNormalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])]))

    wlp300_dataloader = DataLoader(dataset=wlp300, batch_size=FLAGS['batch_size'], shuffle=True, num_workers=1)

    # 2) Intermediate Processing.
    transform_img = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])
    ])

    # 3) Create PRNet model.
    start_epoch, target_epoch = FLAGS['start_epoch'], FLAGS['target_epoch']
    model = ResFCN256(resolution_input=256, resolution_output=256, channel=3, size=16)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS["lr"], betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    stat_loss = SSIM(mask_path=FLAGS["mask_path"], gauss=FLAGS["gauss_kernel"])
    loss = WeightMaskLoss(mask_path=FLAGS["mask_path"])
    bce_loss = torch.nn.BCEWithLogitsLoss()
    bce_loss.to(FLAGS["device"])
    
    #Loss function for adversarial
    for ep in range(start_epoch, target_epoch):
        bar = tqdm(wlp300_dataloader)
        loss_list_G, stat_list = [], []
        for i, sample in enumerate(bar):
            uv_map, origin = sample['uv_map'].to(FLAGS['device']), sample['origin'].to(FLAGS['device'])

            # Inference.
            optimizer.zero_grad()
            uv_map_result = model(origin)
            loss_g = bce_loss(uv_map_result, uv_map)
            loss_g.backward()
            loss_list_G.append(loss_g.item())
            optimizer.step()
        

        if ep % FLAGS["save_interval"] == 0:
            
            with torch.no_grad():
                print(" {} [BCE ({})]".format(ep, loss_list_G[-1]))
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
                'start_epoch': ep
            }
            torch.save(state, os.path.join(FLAGS['images'], '{}.pth'.format(ep)))

            scheduler.step()
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", help="specify input directory.")
    args = parser.parse_args()
    main(args.train_dir)
