from torch.utils.data import Dataset
from datasets.af2he import AF2HEDataset
from datasets.hemit import HEMITDataset
from datasets.cuhk import CUHKDataset
from datasets.aperio import AperioDataset
from datasets.he2pas import HE2PASDataset


CLS = {
    'af2he': AF2HEDataset, 'hemit': HEMITDataset,
    'cuhk': CUHKDataset, 'aperio': AperioDataset,
    'he2pas': HE2PASDataset
}

class UNI(Dataset):
    def __init__(self, dataset_name, dataroot, crop_size, phase, noise):
        self.dataset = CLS[dataset_name](dataroot, crop_size, phase, 0, noise)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        return data