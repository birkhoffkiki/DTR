from PIL import Image
from torch.utils.data import Dataset
import random
from torchvision.transforms import functional
import os
import torch
from datasets.random_affine import get_transformers


class AF2HEDataset(Dataset):
    def __init__(self, data_root, size, phase=None, random_seed=0, noise=0):
        self.random_seed = random_seed
        random.seed(random_seed)
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
        pairs = []
        mode = 'train' if self.phase == 'train' else 'test'
        root = os.path.join(self.data_root, mode)
        slides = os.listdir(root)
        for s in slides:
            slide_dir = os.path.join(root, s)
            patches_dir = os.listdir(slide_dir)
            if '.DS_Store' in patches_dir:
                patches_dir.remove('.DS_Store')
            for dir in patches_dir:
                prefix = os.path.join(slide_dir, dir)
                names = os.listdir(prefix)
                if '.DS_Store' in names:
                    names.remove('.DS_Store')
                # af is 1
                if len(names) != 2:
                    continue
                flag = names[0].split('_')[0]
                if flag == '1':
                    p = [os.path.join(prefix, i) for i in names]
                else:
                    p = [os.path.join(prefix, i) for i in names][::-1]
                pairs.append(p)
        return pairs

    def __len__(self):
        return len(self.image_names)

    def read_img(self, item):
        af_path, he_path = self.image_names[item]
        size = self.size
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
        af, he, name = self.read_img(item)
        # flip
        if self.phase == 'train':
            flip_func = self.flip()
            he = flip_func(he)
            af = flip_func(af)

        he_data = (functional.to_tensor(he) - 0.5)*2
        af_data = (functional.to_tensor(af) - 0.5)*2
        af_data = torch.cat([af_data, af_data, af_data], dim=0)
        
        save_name = os.path.split(name[0])[-1]
        
        data = {
            'input': af_data,
            'gt': he_data,
            'info': self.image_names[item][0],
            'xy': self.image_names[item][1],
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
    dataset = AF2HEDataset('/home/jmabq/data/af2he_uniform', 128, 'train', 0, noise)
    data = dataset[10]
    x = (data['input'] + 1)/2
    y = (data['gt'] + 1)/2
    x = functional.to_pil_image(x)
    y = functional.to_pil_image(y)
    x.save('x.png')
    y.save('y.png')

