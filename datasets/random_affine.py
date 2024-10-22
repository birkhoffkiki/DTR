from torchvision.transforms import functional
from torchvision import transforms as ttv


def get_transformers(noise=0):
    """
    noise level: 0 for zero noise, 5 for max level noise
    """
    # rotate angle
    r = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
    # translation 
    t = {0:0, 1:0.02, 2:0.04, 3:0.06, 4:0.08, 5: 0.1}
    # scale
    s = {0:0, 1:0.02, 2:0.04, 3:0.06, 4:0.08, 5: 0.1}
    affine = ttv.RandomAffine(degrees=r[noise], translate=(t[noise], t[noise]),
                            scale=(1-s[noise], 1+s[noise]), interpolation=functional.InterpolationMode.BILINEAR)
    return affine
