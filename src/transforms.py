from torchvision import transforms, utils
import numpy as np
import PIL
# custom libraries
# fast chirplet transformation
# from fastchirplet import chirplet as ch

def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    """
    Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
    """
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)
    
    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min) # 255
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def transformations(img_size):
    return transforms.Compose([
            transforms.Lambda(lambda img: mono_to_color(img)),
            transforms.ToPILImage(),
            transforms.Resize(size=(img_size, img_size), interpolation=PIL.Image.NEAREST), # NCHW
    #         transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor() # normalizes values to be [0,1]
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # to normalize or not?
            ])