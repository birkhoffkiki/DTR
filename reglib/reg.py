# system
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# local
from .layers import DownBlock, Conv, ResnetTransformer
from .nn import mean_flat


sampling_align_corners = False

# The number of filters in each block of the encoding part (down-sampling).
ndf = {'A': [32, 64, 64, 64, 64, 64], 'B': [32, 64, 64, 64, 64, 64, 64], }
# The number of filters in each block of the decoding part (up-sampling).
# If len(ndf[cfg]) > len(nuf[cfg]) - then the deformation field is up-sampled to match the input size.
nuf = {'A': [64, 64, 64, 64, 64, 32], 'B': [64, 64, 64, 64, 64, 64, 32], }
# Indicate if res-blocks are used in the down-sampling path.
use_down_resblocks = {'A': True, }
# indicate the number of res-blocks applied on the encoded features.
resnet_nblocks = {'A': 3, }
# Indicate if the a final refinement layer is applied on the before deriving the deformation field
refine_output = {'A': True, }
# The activation used in the down-sampling path.
down_activation = {'A': 'leaky_relu', }
# The activation used in the up-sampling path.
up_activation = {'A': 'leaky_relu', }


def smooothing_loss_reg(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    dx = dx*dx
    dy = dy*dy
    d = torch.mean(dx) + torch.mean(dy)
    return d


def smoothing_loss(deformation, img=None, alpha=1.0):
    """Calculate the smoothness loss of the given defromation field
    :param deformation: the input deformation
    :param img: the image that the deformation is applied on (will be used for the bilateral filtering).
    :param alpha: the alpha coefficient used in the bilateral filtering.
    :return:
    adopt from: https://github.com/moabarar/nemar/blob/0333edeb1582d8530c8c39d6d0b0150a30cc214f/models/stn/stn_losses.py
    """
    diff_1 = torch.abs(deformation[:, :, 1:, :] - deformation[:, :, :-1, :])
    diff_2 = torch.abs((deformation[:, :, :, 1:] - deformation[:, :, :, :-1]))
    diff_3 = torch.abs(deformation[:, :, 0:-1, 0:-1] - deformation[:, :, 1:, 1:])
    diff_4 = torch.abs(deformation[:, :, 0:-1, 1:] - deformation[:, :, 1:, 0:-1])
    if img is not None and alpha > 0.0:
        mask = img
        weight_1 = torch.exp(-alpha * torch.abs(mask[:, :, 1::, :] - mask[:, :, 0:-1, :]))
        weight_1 = torch.mean(weight_1, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_2 = torch.exp(- alpha * torch.abs(mask[:, :, :, 1::] - mask[:, :, :, 0:-1]))
        weight_2 = torch.mean(weight_2, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_3 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-1, 0:-1] - mask[:, :, 1::, 1::]))
        weight_3 = torch.mean(weight_3, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_4 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-1, 1::] - mask[:, :, 1::, 0:-1]))
        weight_4 = torch.mean(weight_4, dim=1, keepdim=True).repeat(1, 2, 1, 1)
    else:
        weight_1 = weight_2 = weight_3 = weight_4 = 1.0
    loss = torch.mean(weight_1 * diff_1) + torch.mean(weight_2 * diff_2) \
           + torch.mean(weight_3 * diff_3) + torch.mean(weight_4 * diff_4)
    return loss


def regularization_loss(deformation, img=None, alpha=1.0, level=1):
    """
    multi-scale regularization loss
    """
    dh, dw = deformation.shape[2:]
    img = None if img is None else img.detach()
    reg = 0.0
    factor = 1.0
    for i in range(level):
        if i != 0:
            deformation_resized = F.interpolate(deformation, (dh // (2 ** i), dw // (2 ** i)), mode='bilinear')
            img_resized = F.interpolate(img, (dh // (2 ** i), dw // (2 ** i)), mode='bilinear')
        else:
            deformation_resized = deformation
            img_resized = img
        reg += factor * smoothing_loss(deformation_resized, img_resized, alpha=alpha)
        factor /= 2.0
    return reg


class ResUnet(torch.nn.Module):
    def __init__(self, nc_a, nc_b, cfg, init_func, init_to_identity):
        super(ResUnet, self).__init__()
        act = down_activation[cfg]
        # ------------ Down-sampling path
        self.ndown_blocks = len(ndf[cfg])
        self.nup_blocks = len(nuf[cfg])
        assert self.ndown_blocks >= self.nup_blocks
        in_nf = nc_a + nc_b
        conv_num = 1
        skip_nf = {}
        for out_nf in ndf[cfg]:
            setattr(self, 'down_{}'.format(conv_num),
                    DownBlock(in_nf, out_nf, 3, 1, 1, activation=act, init_func=init_func, bias=True,
                              use_resnet=use_down_resblocks[cfg], use_norm=False))
            skip_nf['down_{}'.format(conv_num)] = out_nf
            in_nf = out_nf
            conv_num += 1
        conv_num -= 1
        if use_down_resblocks[cfg]:
            self.c1 = Conv(in_nf, 2 * in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
            self.t = ((lambda x: x) if resnet_nblocks[cfg] == 0
                      else ResnetTransformer(2 * in_nf, resnet_nblocks[cfg], init_func))
            self.c2 = Conv(2 * in_nf, in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
        # ------------- Up-sampling path
        act = up_activation[cfg]
        for out_nf in nuf[cfg]:
            setattr(self, 'up_{}'.format(conv_num),
                    Conv(in_nf + skip_nf['down_{}'.format(conv_num)], out_nf, 3, 1, 1, bias=True, activation=act,
                         init_fun=init_func, use_norm=False, use_resnet=False))
            in_nf = out_nf
            conv_num -= 1
        if refine_output[cfg]:
            self.refine = nn.Sequential(ResnetTransformer(in_nf, 1, init_func),
                                        Conv(in_nf, in_nf, 1, 1, 0, use_resnet=False, init_func=init_func,
                                             activation=act,
                                             use_norm=False)
                                        )
        else:
            self.refine = lambda x: x
        self.output = Conv(in_nf, 2, 3, 1, 1, use_resnet=False, bias=True,
                           init_func=('zeros' if init_to_identity else init_func), activation=None,
                           use_norm=False)
    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        skip_vals = {}
        conv_num = 1
        # Down
        while conv_num <= self.ndown_blocks:
            x, skip = getattr(self, 'down_{}'.format(conv_num))(x)
            skip_vals['down_{}'.format(conv_num)] = skip
            conv_num += 1
        if hasattr(self, 't'):
            x = self.c1(x)
            x = self.t(x)
            x = self.c2(x)
        # Up
        conv_num -= 1
        while conv_num > (self.ndown_blocks - self.nup_blocks):
            s = skip_vals['down_{}'.format(conv_num)]
            x = F.interpolate(x, (s.size(2), s.size(3)), mode='bilinear')
            x = torch.cat([x, s], 1)
            x = getattr(self, 'up_{}'.format(conv_num))(x)
            conv_num -= 1
        x = self.refine(x)
        x = self.output(x)
        return x


class Reg(nn.Module):
    def __init__(self, height, width, in_channels_a,in_channels_b, device, init_to_identity=True):
        super(Reg, self).__init__()
        #height,width=256,256
        #in_channels_a,in_channels_b=1,1
        init_func = 'kaiming'
        # paras end------------
        self.oh, self.ow = height, width
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.device = device 
        self.offset_map = ResUnet(self.in_channels_a, self.in_channels_b, cfg='A',
                             init_func=init_func, init_to_identity=init_to_identity).to(device)
        self.identity_grid = self.get_identity_grid()

    def get_identity_grid(self):
        x = torch.linspace(-1.0, 1.0, self.ow)
        y = torch.linspace(-1.0, 1.0, self.oh)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return identity

    def forward(self, img_a, img_b, apply_on=None):
        deformations = self.offset_map(img_a, img_b)
        return deformations


class Transformer_2D(nn.Module):
    """'
    apply transformation
    """
    def __init__(self):
        super(Transformer_2D, self).__init__()

    def forward(self,src, flow):
        b = flow.shape[0]
        h = flow.shape[2]
        w = flow.shape[3]

        size = (h,w)

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b,1,1,1).cuda()
        new_locs = grid+flow
        shape = flow.shape[2:]
        # print('flow max:', flow.max())
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1 , 0]]
        warped = F.grid_sample(src,new_locs,align_corners=True,padding_mode="border")
        # get mask
        x = (new_locs[..., 0] <=1) * (new_locs[..., 0] >= -1)
        y = (new_locs[..., 1] <=1) * (new_locs[..., 1] >= -1)
        mask = x*y
        mask = mask[:, None]
        # ctx.save_for_backward(src,flow)
        return warped, mask


class Transformer_2DReg(nn.Module):
    """
    https://github.com/Kid-Liet/Reg-GAN/blob/main/trainer/transformer.py
    """
    def __init__(self):
        super(Transformer_2DReg, self).__init__()
    # @staticmethod
    def forward(self,src, flow):
        b = flow.shape[0]
        h = flow.shape[2]
        w = flow.shape[3]

        size = (h,w)

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b,1,1,1).cuda()
        new_locs = grid+flow
        shape = flow.shape[2:]
        # print('flow max:', flow.max())
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1 , 0]]
        warped = F.grid_sample(src,new_locs,align_corners=True,padding_mode="border")
        # ctx.save_for_backward(src,flow)
        return warped


class RegLoss:
    def __init__(self, device, T, SReg, DReg=None, mode='couple', height=128, width=128) -> None:
        self.T = T
        self.SReg = SReg
        self.DReg = DReg
        self.mode = mode
        if mode == 'couple':
            # Reg-GAN transforme
            self.spatial_transform = Transformer_2DReg()
        else:
            self.spatial_transform = Transformer_2D(device, width, height)

    def __call__(self, input, target, cfg):
        if self.mode == 'decouple':
            mesh1 = self.DReg(input, target)
            source1 = self.spatial_transform(input, mesh1)
            rec1 = self.T(source1)

            # trans first and then register
            rec2 = self.T(input)
            mesh2 = self.SReg(rec2, target)
            rec2_reg = self.spatial_transform(rec2, mesh2)
            
            # add loss
            loss_regis_error = torch.abs(rec2 - rec2_reg).mean()
            l11 = F.l1_loss(rec1, target)*cfg.lambda_l1
            l12 = F.l1_loss(rec2_reg, target)*cfg.lambda_l1

            mesh_smooth = (regularization_loss(mesh1, input) + regularization_loss(mesh2, target))*cfg.weight_smooth
            mesh_sim = F.l1_loss(mesh1, mesh2)*cfg.weight_mesh_sim
            total_loss = l11 + l12 + mesh_smooth + mesh_sim
            loss = {'total_loss': total_loss,'l11': l11, 'l12': l12, 'regis_error': loss_regis_error, 
                    'mesh_smooth': mesh_smooth, 'mesh_sim': mesh_sim}
            images = {'source1': source1, 'rec1': rec1, 'rec2': rec2, 'rec2_reg':rec2_reg}
            return loss, images
        
        elif self.mode == 'couple':
            rec = self.T(input)
            mesh = self.SReg(rec, target)
            rec_reg = self.spatial_transform(rec, mesh)
            l1_loss = F.l1_loss(rec_reg, target)*cfg.lambda_l1
            mesh_smooth = smooothing_loss_reg(mesh)*cfg.weight_smooth
            total_loss = l1_loss + l1_loss
            loss = {'total_loss': total_loss, 'l1_loss': l1_loss, 'mesh_smooth': mesh_smooth}
            images = {'rec': rec,'rec_reg': rec_reg}
            return loss, images
        else:
            raise RuntimeError(f'{self.mode} is not supported.')