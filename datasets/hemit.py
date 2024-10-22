from PIL import Image
from torch.utils.data import Dataset
import random
from torchvision.transforms import functional
import os
from datasets.random_affine import get_transformers


class HEMITDataset(Dataset):
    def __init__(self, data_root, size, phase=None, random_seed=0, noise=None):
        self.random_seed = random_seed
        random.seed(random_seed)
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.data_root = data_root
        self.size = size
        self.noise = noise
        self.image_names = self.parse_files()
        if noise in [1, 2, 3, 4, 5]:
            print('DATA tranformation is adopt, noise level:', noise)
            self.affine = get_transformers(noise=noise)
        else:
            self.noise = None
        

    def parse_files(self):
        pairs = []
        root = os.path.join(self.data_root, self.phase)
        names = os.listdir(os.path.join(root, 'input'))
        for n in names:
            input_p = os.path.join(root, 'input', n)
            gt_p = os.path.join(root, 'label', n)
            pairs.append((input_p, gt_p))
        return pairs

    def __len__(self):
        return len(self.image_names)

    def read_img(self, item):
        af_path, he_path = self.image_names[item]
        size = self.size
        # print(af_path, he_path)
        af = Image.open(af_path).convert('RGB')
        he = Image.open(he_path).convert('RGB')
        if self.phase == 'train':
            x = random.randint(0, 1024-size)
            y = random.randint(0, 1024-size)
        else:
            x = int((1024 - size)//2)
            y = x

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
    dataset = HEMITDataset('/home/jmabq/data/HEMIT', 512, 'train', 0, noise)
    data = dataset[10]
    x = (data['input'] + 1)/2
    y = (data['gt'] + 1)/2
    x = functional.to_pil_image(x)
    y = functional.to_pil_image(y)
    x.save('x.png')
    y.save('y.png')

