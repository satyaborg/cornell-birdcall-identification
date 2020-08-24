from torchvision import transforms, utils
import numpy as np
import PIL
# custom libraries
# fast chirplet transformation
# from fastchirplet import chirplet as ch
# https://stats.stackexchange.com/questions/426818/do-i-need-3-rgb-channels-for-a-spectrogram-cnn
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

def transformations(img_size, sr, duration, hop_length):
    """Transformations and augmentations
    if (duration*sr i.e. 5 dec chunk) > length of clip, then random pad with zeros  
    """

    crop_width = duration * (sr // hop_length)
    return transforms.Compose([
            transforms.Lambda(lambda img: mono_to_color(img)),
            transforms.ToPILImage(),
            transforms.RandomCrop(size=(128, crop_width), pad_if_needed=True), # randomly crops 5 secs
            # NCHW; remove one img_size to make it (224x542)
            transforms.Resize(size=(img_size, img_size), interpolation=PIL.Image.NEAREST), 
            transforms.ToTensor(), # normalizes values to be [0,1]
            # transforms.Normalize(mean=mean, std=std)
            ])

def spec_augment():
    """Apply SpecAugment to each clip
    """

    pass

def mixup():

    pass

def add_noise():
    pass


def extract_sec_labels():
    pass