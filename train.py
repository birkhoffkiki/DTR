from __future__ import print_function
import os
from time import time
from batch_utils import UNI
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import pytorch_ssim
import Attention_GAN
from metrics import easy_psnr
from tqdm import tqdm
from reglib import reg
import argparse


def check_path(path):
    if not os.path.exists(path):
        print('creating:{}'.format(path))
        os.makedirs(path)
    return path


def get_generator(args):
    pass


def parallel_wrapper(model, gpu_id: str):
    if len(gpu_id.split(',')) > 1:
        model = nn.DataParallel(model)
    return model


def load_ckpt(netG, netD, RegGT, RegX, weight_root, weight_id):
    print('Loading pretrained checkpoint:{}'.format(weight_id))
    path = os.path.join(weight_root, 'netG_epoch_{}.pth'.format(weight_id))
    msg = netG.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
    print('netG:', msg)

    path = os.path.join(weight_root, 'netD_epoch_{}.pth'.format(weight_id))
    msg = netD.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
    print('netD:', msg)

    path = os.path.join(weight_root, 'RegGT_epoch_{}.pth'.format(weight_id))
    msg = RegGT.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
    print('RegGT:', msg)

    path = os.path.join(weight_root, 'RegX_epoch_{}.pth'.format(weight_id))
    msg = RegX.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
    print('RegX:', msg)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help="the dataset name")
    parser.add_argument('--backbone', type=str, help="the backbone name")
    parser.add_argument('--data_root', type=str, help="The data root of the data")
    parser.add_argument('--crop_size', type=int, help='the size of cropped images')
    parser.add_argument('--phase', type=str, help='train, val, test')
    parser.add_argument('--noise', type=int, help='the noise level, 0, 1, 2, 3, 4, 5')
    parser.add_argument('--n_epochs', type=int, help='training epochs')
    parser.add_argument('--save_per_epoch', type=int, default=1, help='saving checkpoint per xx epoch')
    parser.add_argument('--batch_size', type=int, help='training batch_size')
    parser.add_argument('--gpu_id', type=str, help='GPU ID')
    parser.add_argument('--checkpoint_dir', type=str, help='path to save checkpoints')
    parser.add_argument('--continue_weight_id', type=int, default=0, help='continue training')

    parser.add_argument('--update_g_freq', type=int, default=4)
    parser.add_argument('--n_channel', type=int, default=64)
    parser.add_argument('--in_channel_num', type=int, default=3)
    parser.add_argument('--out_channel_num', type=int, default=3)
    parser.add_argument('--lr_G', type=float, default=1e-4, help="The initial lr of G")
    parser.add_argument('--lr_D', type=float, default=1e-5, help="The initial lr of D")
    parser.add_argument('--lr_G_reg', type=float, default=1e-5, help="The initial lr of G Reg")
    parser.add_argument('--lr_D_reg', type=float, default=1e-5, help="The initial lr of D Reg")

    # loss balances
    parser.add_argument('--rec_weight', type=float, default=10, help="weight of rec_after_reg and gt")
    parser.add_argument('--GAN_weight', type=float, default=0.5, help="weight of GAN loss")
    parser.add_argument('--smooth_weight', type=float, default=10, help="weight of registration smooth")
    parser.add_argument('--rec_ssim_weight', type=float, default=0.5, help="ssim loss")
    parser.add_argument('--input_mesh_weight', type=float, default=5, help="the mesh field between the input and the reconstruced image")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # ------- extra configs-----------------
    init_to_identity = True  # deformation field
    decouple_iteration = 100
    # --------------------------------------

    # define dataset
    train_set = UNI(args.dataset_name, args.data_root, args.crop_size, 'train', args.noise)
    valid_set = UNI(args.dataset_name, args.data_root, args.crop_size, 'test', args.noise)
    train_data_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=args.batch_size,
                        shuffle=True, drop_last=False, pin_memory=True)
    valid_data_loader = DataLoader(dataset=valid_set, num_workers=8, batch_size=args.batch_size,
                        shuffle=False, drop_last=False)

    im_save_path = check_path(os.path.join(args.checkpoint_dir, 'validation_images'))
    model_save_path = check_path(os.path.join(args.checkpoint_dir, 'model'))
    summary_path = os.path.join(args.checkpoint_dir, 'summary.log')

    print("===> Building model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # build generator and discriminator
    netG = Attention_GAN.Generator(n_channels=args.n_channel, in_channels=args.in_channel_num, batch_norm=False,
        out_channels=args.out_channel_num, padding=1, pooling_mode="maxpool",).to(device)
    netD = Attention_GAN.Discriminator(n_channels=args.n_channel, in_channels=args.in_channel_num, batch_norm=False).to(device)

    # define models for registration
    RegGT = reg.Reg(args.crop_size, args.crop_size, 3, 3, device, init_to_identity)
    RegX = reg.Reg(args.crop_size, args.crop_size, 3, 3, device, init_to_identity) # predict mesh field
    spatial_transform = reg.Transformer_2D()

    # load pretrained model
    if args.continue_weight_id > 0:
        load_ckpt(netG, netD, RegGT, RegX, model_save_path, args.continue_weight_id)

    writer = open(summary_path, 'w+')
    # loss functions
    bce_loss_fun = nn.BCEWithLogitsLoss(size_average=True).to(device)
    l1_loss_fun = nn.SmoothL1Loss(size_average=True).to(device)
    ssim_loss_fun = pytorch_ssim.SSIM(size_average=True).to(device)

    # setup optimizer
    optimizerRegGT = optim.AdamW(RegGT.parameters(), lr=args.lr_G_reg, betas=(0.9, 0.999)) # RegNet for target style
    optimizerRegX = optim.AdamW(RegX.parameters(), lr=args.lr_D_reg, betas=(0.9, 0.999)) # RegNet for input style
    optimizerD = optim.AdamW(netD.parameters(), lr=args.lr_D, betas=(0.9, 0.999))
    optimizerG = optim.AdamW(netG.parameters(), lr=args.lr_G, betas=(0.9, 0.999))

    netG = parallel_wrapper(netG, args.gpu_id)
    netD = parallel_wrapper(netD, args.gpu_id)
    RegGT = parallel_wrapper(RegGT, args.gpu_id)
    RegX = parallel_wrapper(RegX, args.gpu_id)

    counter = 0
    save_img_freq = int(len(train_data_loader)//100)
    best_psnr = 0
    for epoch in range(args.continue_weight_id+1, args.n_epochs+1):
        start_time = time()
        # train
        netG.train()
        netD.train()
        RegGT.train()
        RegX.train()
        for i, batch in enumerate(train_data_loader):
            counter += 1
            x = (batch['input']).to(device)
            gt = (batch['gt']).to(device)

            ###########################
            # (1) Update G
            ###########################
            rec = netG(x)
            mesh = RegGT(rec, gt)
            rec_after_reg, mask = spatial_transform(rec, mesh)

            # loss items after rec
            G_rec_loss = l1_loss_fun(rec_after_reg*mask, gt*mask)*args.rec_weight
            G_mesh_loss = reg.smooothing_loss_reg(mesh)*args.smooth_weight
            G_ssim_loss = -torch.log((1 + ssim_loss_fun((rec_after_reg*mask + 1.0)*0.5, (gt*mask + 1.0)*0.5)) / 2)*args.rec_ssim_weight

            # the deformation field between x and reconstuced image. The deformation field should be identity mapping.
            if counter >= decouple_iteration:
                same_mesh = RegX(x, rec)
                same_reg, s_mask = spatial_transform(rec, same_mesh)
                G_same_mesh = l1_loss_fun(rec*s_mask, same_reg*s_mask)*args.input_mesh_weight
            else:
                G_same_mesh = 0

            # GAN loss
            fake_label = netD(rec)
            G_dis_loss = bce_loss_fun(fake_label, torch.ones_like(fake_label))*args.GAN_weight

            # --------------Update netG and RegGT model----------------
            errG = G_dis_loss + G_ssim_loss + G_mesh_loss + G_rec_loss + G_same_mesh
            RegGT.zero_grad()
            netG.zero_grad()
            clip_grad_norm_(netG.parameters(), 0.5)
            clip_grad_norm_(RegGT.parameters(), 0.5)
            errG.backward()
            optimizerG.step()
            optimizerRegGT.step()
            # ----------------------------------------------------------

            #================================
            # update RegX
            #================================
            if counter >= decouple_iteration:
                RegX.zero_grad()
                rec = rec.detach()
                same_mesh = RegX(x, rec)
                same_reg, ds_mask = spatial_transform(rec, same_mesh)
                same_mesh_err = l1_loss_fun(rec, same_reg)

                diff_mesh = RegX(x, gt)
                diff_reg, dd_mask = spatial_transform(rec, diff_mesh)
                diff_mesh_err = l1_loss_fun(diff_reg,  rec_after_reg.detach())

                D_same_mesh_loss = reg.smooothing_loss_reg(diff_mesh)*args.smooth_weight
                D_diff_mesh_loss = reg.smooothing_loss_reg(same_mesh)*args.smooth_weight

                G_netDReg_loss = same_mesh_err + diff_mesh_err + D_same_mesh_loss + D_diff_mesh_loss
                G_netDReg_loss.backward()
                optimizerRegX.step()
            else:
                same_mesh_err = 0
                diff_mesh_err = 0

            ############################
            # (2) Update D network, the freq is args.update_g_freq
            ###########################
            indie_G = args.update_g_freq
            if i % indie_G == 0:
                netD.zero_grad()
                fake_label = netD(rec.detach())
                D_fake_loss = bce_loss_fun(fake_label, torch.zeros_like(fake_label))
                D_real_loss = bce_loss_fun(netD(gt), torch.ones_like(fake_label))
                errD = (D_fake_loss + D_real_loss) * 0.5
                clip_grad_norm_(netD.parameters(), 0.5)
                errD.backward()
                optimizerD.step()
                log_text = 'Epoch: [{}/{}], Iteration: {}, G_total:{:.4f}, G_l1_after_reg:{:.4f}, G_reg_mesh: {:.4f}, G_ssim_after_reg: {:.4f}, G_same_mesh: {:.4f}, G_gan:{:.4f}, RegD: {:.4f}, RegG:{:.4f}, D_fake:{:.4f}, D_real:{:.4f}'.format(
                      epoch, args.n_epochs+1, counter, errG, G_rec_loss, G_mesh_loss, G_ssim_loss, G_same_mesh, G_dis_loss,
                      same_mesh_err, diff_mesh_err, D_fake_loss, D_real_loss)
                print(log_text)
                writer.write(log_text+'\n')

            if counter % save_img_freq == 0 and counter >= decouple_iteration:
                err1 = (torch.abs(rec[0]-rec_after_reg[0]) - 0.5)*2
                tmp = [x[0], rec[0], rec_after_reg[0], err1, gt[0], same_reg[0], diff_reg[0], mask[0].expand(3, -1, -1)]
                tmp = torch.cat(tmp, dim=2)
                vutils.save_image(tmp, os.path.join(im_save_path, "save_epoch_{}_{}.png".format(epoch, counter)), normalize=True, value_range=(-1, 1),)

        if epoch % args.save_per_epoch == 0:
            print('Start evaluating ...')
            # valid
            val_loss_G_ssim = 0
            test_counter = 0
            netG.eval()
            psnr_list = []
            with torch.no_grad():
                for i, batch in tqdm(enumerate(valid_data_loader)):
                    test_counter += 1
                    val_input = (batch['input']).to(device)
                    val_target = (batch['gt']).to(device)
                    val_fake = netG(val_input)
                    # compute psnr
                    predict_list = []
                    gt_list = []
                    for j in range(len(val_target)):
                        predict_list.append(val_fake[j])
                        gt_list.append(val_target[j])
                    predict_list = torch.stack(predict_list, dim=0)
                    gt_list = torch.stack(gt_list, dim=0)
                    # convert (-1, 1) to (0, 1)
                    predict_list = (predict_list + 1)/2
                    gt_list = (gt_list + 1)/2
                    _psnr = easy_psnr(predict_list, gt_list)
                    psnr_list.append(_psnr)
            temp_psnr = sum(psnr_list)/len(psnr_list)
            best_psnr = temp_psnr if temp_psnr > best_psnr else best_psnr
            writer.write('Evaluating, Epoch [{}], PSNR:{:.3f}, Best PSNR: {:.3f}\n'.format(epoch, temp_psnr, best_psnr))

            torch.save(netG.state_dict(), os.path.join(model_save_path, 'netG_epoch_{}.pth'.format(epoch)))
            torch.save(netD.state_dict(), os.path.join(model_save_path, 'netD_epoch_{}.pth'.format(epoch)))
            torch.save(RegGT.state_dict(), os.path.join(model_save_path, 'RegGT_epoch_{}.pth'.format(epoch)))
            torch.save(RegX.state_dict(), os.path.join(model_save_path, 'RegX_epoch_{}.pth'.format(epoch)))
            