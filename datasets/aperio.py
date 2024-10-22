from PIL import Image
from torch.utils.data import Dataset
import random
from torchvision.transforms import functional
import os
from datasets.random_affine import get_transformers

from PIL import Image
import random
from torchvision.transforms import functional
import os


class AperioDataset(Dataset):
    def __init__(self, data_root, size, phase=None, random_seed=0, noise=None):
        random_seed = 0
        self.size = size
        self.noise = noise
        random.seed(random_seed)
        self.data_root = data_root

        self.phase = phase if phase == 'train' else 'test'   # The behaviour of val and test is same
        path = os.path.join(self.data_root, self.phase, 'aperio')
        self.image_names = [i for i in os.listdir(path) if '.png' in i]
        if noise in [1, 2, 3, 4, 5]:
            print('DATA tranformation is adopt, noise level:', noise)
            self.affine = get_transformers(noise=noise)
        else:
            self.noise = None
         

    def __len__(self):
        return len(self.image_names)

    def read_img(self, name):
        size = self.size
        af_path = os.path.join(self.data_root, self.phase, 'aperio', name)
        he_path = os.path.join(self.data_root, self.phase, 'hamamatsu', 'H' + name[1:]) # val is from train
        af = Image.open(af_path)
        he = Image.open(he_path)
        
        if self.phase == 'train':
            x = random.randint(0, 256-size)
            y = random.randint(0, 256-size)
        else:
            x, y = 64, 64

        # transform af image for ablation study, only for train
        if self.noise is not None and self.phase == 'train':
            af = self.affine(af)
            
        he = functional.crop(he, x, y, size, size)
        af = functional.crop(af, x, y, size, size)

        return af, he, (af_path, he_path)

    def __getitem__(self, item):
        name = self.image_names[item]
        # he is aperio, ihc is hamamatsu
        he, ihc, (a_path, b_path) = self.read_img(name)
        # flip
        if self.phase == 'train':
            flip_func = self.flip()
            he = flip_func(he)
            ihc = flip_func(ihc)

        he_data = (functional.to_tensor(he) - 0.5)*2
        ihc_data = (functional.to_tensor(ihc) - 0.5)*2
        
        save_name = os.path.split(a_path)[-1]
        
        data = {
            'input': he_data,
            'gt': ihc_data,
            'name': save_name,
        }
        return data

    @staticmethod
    def flip():
        h = random.random()
        v = random.random()        
        def func(input_data):
            input_data = functional.hflip(input_data) if h > 0.5 else input_data
            input_data = functional.vflip(input_data) if v > 0.5 else input_data
            return input_data
        return func



if __name__ == '__main__':
    noise = 2
    dataset = AperioDataset('/home/jmabq/data/aperio_hamamatsu', 128, 'train', 0, noise)
    data = dataset[10]
    x = (data['input'] + 1)/2
    y = (data['gt'] + 1)/2
    x = functional.to_pil_image(x)
    y = functional.to_pil_image(y)
    x.save('x.png')
    y.save('y.png')

