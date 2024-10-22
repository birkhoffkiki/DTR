from PIL import Image
from torch.utils.data import Dataset
import random
from torchvision.transforms import functional
import os
import json
from datasets.random_affine import get_transformers


class CUHKDataset(Dataset):
    """
    large-scale HE2PAS dataset, default: 256*256
    """
    def __init__(self, data_root, size=128, phase=None, random_seed=0, noise=None):
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.phase = phase if phase == 'train' else 'val'   # The behaviour of val and test is same
        self.data_root = data_root
        self.size = size
        self.image_names = self.parse_files()
        self.noise = noise
        if noise in [1, 2, 3, 4, 5]:
            print('DATA tranformation is adopt, noise level:', noise)
            self.affine = get_transformers(noise=noise)
        else:
            self.noise = None
        


    def parse_files(self):
        json_file = 'train.json' if self.phase == 'train' else 'test.json'
        p = os.path.join(self.data_root, json_file)
        with open(p) as f:
            image_names = json.load(f)['items']
        return image_names

    def __len__(self):
        return len(self.image_names)

    def read_img(self, item):
        a, b = [os.path.join(self.data_root, i) for i in  self.image_names[item]]
        size = self.size
        img_a = Image.open(a).convert('RGB')
        img_b = Image.open(b).convert('RGB')


        # transform af image for ablation study, only for train
        if self.noise is not None and self.phase == 'train':
            img_a = self.affine(img_a)
            

        if self.phase == 'train':
            x = random.randint(0, 256-size)
            y = random.randint(0, 256-size)
        else:
            y = x = int((256 - size)//2)

        img_a = functional.crop(img_a, x, y, size, size)
        img_b = functional.crop(img_b, x, y, size, size)

        a_n, b_n = self.image_names[item]
        a_n = '__'.join(a_n.split('/'))
        b_n = '__'.join(b_n.split('/'))
        
        return img_a, img_b, (a_n, b_n)

    def __getitem__(self, item):
        a, b, name = self.read_img(item)
        # flip
        if self.phase == 'train':
            flip_func = self.flip()
            a = flip_func(a)
            b = flip_func(b)

        a = (functional.to_tensor(a) - 0.5)*2
        b = (functional.to_tensor(b) - 0.5)*2

        data = {
            'input': a,
            'gt': b,
            'name': name[0],
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
    dataset = CUHKDataset('/jhcnas3/VirtualStaining/Patches/HE2PAS_256', 128, 'train', 0, noise=noise)
    data = dataset[22]
    x = (data['input'] + 1)/2
    y = (data['gt'] + 1)/2
    x = functional.to_pil_image(x)
    y = functional.to_pil_image(y)
    x.save('x.png')
    y.save('y.png')

