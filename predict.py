from __future__ import print_function
import os
from batch_utils import UNI
import torch
from torch.utils.data import DataLoader

import cv2
import Attention_GAN
import argparse


def check_path(path):
    if not os.path.exists(path):
        print('creating:{}'.format(path))
        os.makedirs(path)
    return path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help="the dataset name")
    parser.add_argument('--data_root', type=str, help="The data root of the data")
    parser.add_argument('--crop_size', type=int, help='the size of cropped images')
    parser.add_argument('--phase', type=str, help='train, val, test')
    parser.add_argument('--noise', type=int, help='the noise level, 0, 1, 2, 3, 4, 5')
    parser.add_argument('--batch_size', type=int, help='training batch_size')
    parser.add_argument('--gpu_id', type=str, help='GPU ID')
    parser.add_argument('--checkpoint_path', type=str, help='path to save checkpoints')
    parser.add_argument('--results_dir', type=str, help='path to save checkpoints')

    parser.add_argument('--n_channel', type=int, default=64)
    parser.add_argument('--in_channel_num', type=int, default=3)
    parser.add_argument('--out_channel_num', type=int, default=3)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # define dataset
    valid_set = UNI(args.dataset_name, args.data_root, args.crop_size, 'test', args.noise)
    valid_data_loader = DataLoader(dataset=valid_set, num_workers=8, batch_size=args.batch_size, 
                        shuffle=False, drop_last=False)
    im_save_path = check_path(args.results_dir)

    print("===> Building model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    # build generator and discriminator
    netG = Attention_GAN.Generator(n_channels=args.n_channel, in_channels=args.in_channel_num, batch_norm=False, 
        out_channels=args.out_channel_num, padding=1, pooling_mode="maxpool",).to(device)
    msg = netG.load_state_dict(torch.load(args.checkpoint_path), strict=True)
    print(msg)
    netG.eval()
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            print('Progress: [{}/{}]'.format(i, len(valid_data_loader)))
            
            val_input = (batch['input']).to(device)
            val_target = (batch['gt']).to(device)
            name = batch['name']

            val_fake = netG(val_input)
            val_fake = ((val_fake + 1)/2).cpu()
            val_target = ((val_target + 1)/2).cpu()
            val_input = ((val_input + 1)/2).cpu()
            # save images;
            for i in range(len(val_target)):
                path = os.path.join(im_save_path, name[i])
                
                img = torch.cat([val_input[i], val_fake[i], val_target[i]], dim=-1)
                img = img.permute(1, 2, 0).clip_(0, 1.0).numpy()*255
                cv2.imwrite(path, img.astype('uint8')[:, :, ::-1])
    print('Done')
