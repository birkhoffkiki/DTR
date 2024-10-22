import torch
from torch import optim
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel


class TrainHP:
    """
    Training hyper parameter
    """
    lr = 1e-4
    weight_decay = 0.05
    # SRegnet
    lr_sregnet = 1e-4
    weight_decay_sregnet = 0.05
    # DRegnet
    lr_dregnet = 1e-4
    weight_decay_dregnet = 0.05

    # loss weights
    mesh_smooth_ratio = 10
    mesh_loss_ratio = 20
    rec_loss_ratio = 1


class TransModel:
    def __init__(self, model, use_gan=False) -> None:
        self.model = model
        self.use_gan = use_gan
    
    def predict(self, source):
        out = self.model(source)
        return out
    
    def G_loss(self, rec, target):
        """
        get the loss of other methods
        """
        return {'rec_loss':0.0}
    
    def D_loss(self, rec, target):
        return {'d_loss':0.0}


class RegTrainner:
    def __init__(self, opt: TrainHP, T: TransModel, SRegNet: torch.nn.Module = None, DRegNet: torch.nn.Module = None,
                 spatial_transform = None, ddp=False,
                 ):
        """
        T: virtual staining model
        SRegNet: Registration network, it recieves virtually stained and target images
        DRegNet: Registration network, it recieves source and target images
        """
        self.opt = opt
        self.T = T
        self.SRegNet = SRegNet
        self.DRegNet = DRegNet
        self.spatial_transform = spatial_transform
        self.opt_T = optim.AdamW(T.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.opt_SRegNet = optim.AdamW(SRegNet.parameters(), lr=opt.lr_sregnet, weight_decay=opt.weight_decay_sregnet)
        self.opt_DRegNet = optim.AdamW(DRegNet.parameters(), lr=opt.lr_dregnet, weight_decay=opt.weight_decay_dregnet)
        # use ddp?

    def compute_loss(self, source, target):
        # first register and then translate
        mesh1 = self.DRegNet(source, target)
        source1 = self.spatial_transform(source, mesh1)

        rec1 = self.T.predict(source=source1)
        rec_loss_1 = self.T.G_loss(rec=rec1, target=target)

        # first translate and then register
        rec2 = self.T.predict(source=source)
        mesh2 = self.SRegNet(rec2, target)
        rec2_reg = self.spatial_transform(rec2, mesh2)
        rec_loss_2 = self.T.G_loss(rec=rec2_reg, target=target)

        # define mesh smooth regularization
        mesh1_smooth = self.smooothing_loss(mesh1)
        mesh2_smooth = self.smooothing_loss(mesh2)

        # mesh similarity
        mesh_loss = torch.nn.functional.l1_loss(mesh1, mesh2)*self.opt.mesh_loss_ratio

        rec_loss = (rec_loss_1['rec_loss'] + rec_loss_2['rec_loss'])*self.opt.rec_loss_ratio
        mesh_smooth = (mesh1_smooth + mesh2_smooth)*self.opt.mesh_smooth_ratio

        total = rec_loss + mesh_smooth + mesh_loss
        out = {
            'loss': total, 'rec_loss': rec_loss, 'mesh_loss': mesh_loss,
            'mesh_smooth': mesh_smooth
        } + {k+'_1': v for k, v in rec_loss_1.items()} + {k+'_2': v for k, v in rec_loss_2.items()}
        return out
        

    def update_g(self, loss: torch.Tensor):
        loss.backward()
        self.opt_T.step()
        self.opt_SRegNet.step()
        self.opt_DRegNet.step()

    def set_lr(self, lr, lrd, lrs):
        for param in self.opt_T.param_groups:
            param['lr'] = lr
        for param in self.opt_DRegNet.param_groups:
            param['lr'] = lrd
        for param in self.opt_SRegNet.param_groups:
            param['lr'] = lrs


    @staticmethod
    def smooothing_loss(y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        dx = dx*dx
        dy = dy*dy
        d = (dx+dy).mean()
        return d
