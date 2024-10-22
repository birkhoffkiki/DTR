import os
import torch
import torch.nn.parallel
import Attention_GAN
from reglib import reg
from PIL import Image
import numpy as np


p1 = '/home/jmabq/projects/VirtualStainingSOTA/Our/assets/0.png'
p2 = '/home/jmabq/projects/VirtualStainingSOTA/Our/assets/4.png'

g_path = '/jhcnas1/jmabq/virtual_staining_sota/Our/cuhk/model/netG_epoch_4.pth'
reg_gt_path = '/jhcnas1/jmabq/virtual_staining_sota/Our/cuhk/model/RegGT_epoch_4.pth'
reg_x_path = '/jhcnas1/jmabq/virtual_staining_sota/Our/cuhk/model/RegX_epoch_4.pth'

model_save_path = '/home/jmabq/projects/VirtualStainingSOTA/Our/assets'
size = 128
if __name__ == "__main__":
    device = torch.device('cuda')
    img1 = np.array(Image.open(p1)).transpose(2, 0, 1).astype('float32')/255.
    img2 = np.array(Image.open(p2)).transpose(2, 0, 1).astype('float32')/255.
    
    x = (torch.from_numpy(img1)[None].to(device) - 0.5)*2
    gt = (torch.from_numpy(img2)[None].to(device) - 0.5)*2


    # build generator and discriminator
    netG = Attention_GAN.Generator(n_channels=64, in_channels=3, batch_norm=False, out_channels=3, padding=1, pooling_mode="maxpool",).to(device)
    msg = netG.load_state_dict(torch.load(g_path), strict=True)
    print(msg)
    
    # define models for registration
    RegGT = reg.Reg(size, size, 3, 3, device, True)
    msg = RegGT.load_state_dict(torch.load(reg_gt_path), strict=True)
    print(msg)
    RegX = reg.Reg(size, size, 3, 3, device, True) # predict mesh field
    msg = RegX.load_state_dict(torch.load(reg_x_path), strict=True)
    print(msg)
    
    spatial_transform = reg.Transformer_2D()
    netG.eval()
    RegGT.eval()
    RegX.eval()
    
    with torch.no_grad():
        rec = netG(x)
        mesh = RegGT(rec, gt).cpu().numpy()
        same_mesh = RegX(x, rec).cpu().numpy()
        diff_mesh = RegX(x, gt).cpu().numpy()
        np.save(os.path.join(model_save_path, 'mesh.npy'), mesh)
        np.save(os.path.join(model_save_path, 'same_mesh.npy'), same_mesh)
        np.save(os.path.join(model_save_path, 'diff_mesh.npy'), diff_mesh)
            