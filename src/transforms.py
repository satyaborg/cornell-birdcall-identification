from torchvision import transforms, utils
import numpy as np
import random
import math
import PIL
# custom libraries
# fast chirplet transformation
# from fastchirplet import chirplet as ch
# https://stats.stackexchange.com/questions/426818/do-i-need-3-rgb-channels-for-a-spectrogram-cnn
def mono_to_color(X,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    """
    Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
    """
    X = np.array(X)
    # Stack X as [X,X,X]
    # needed for pretrained models
    X = np.stack([X, X, X], axis=-1)
    
    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    # print(_min, _max, Xstd.mean(), Xstd.std())
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
    # print(V.max(), V.min(), V.mean(), V.std())
    return V

def transformations(img_size, duration, mel_params, img_stats):
    """Transformations and augmentations
    crop, resize, normalize/standardize, specaug
    if (duration*sr i.e. 5 dec chunk) > length of clip, then random pad with zeros  
    """
    crop_width = duration * (mel_params.get("sr") // mel_params.get("hop_length"))
    mask_percentage = 0.075
    return dict(
            train=transforms.Compose([
            # transforms.Lambda(lambda img: extend_clip(img, crop_width)), # extend clip < 5 secs
            transforms.ToPILImage(), # covert to PIL image
            transforms.RandomCrop(size=(mel_params.get("n_mels"), crop_width), 
                                pad_if_needed=True, 
                                fill=mel_params.get("min_db")
                                ), # randomly crops 5 secs
            transforms.Resize(size=(img_size, img_size), 
                            interpolation=PIL.Image.NEAREST
                            ), # resize image to be [img_size, img_size]
            transforms.Lambda(lambda x: mono_to_color(x)), # normalize to be [0-255]
            # Apply SpecAugment
            # (height, width, channels) -> (224, 224, 3)
            transforms.Lambda(lambda x:
                        spec_augment(x, num_mask=2, 
                            freq_masking_max_percentage=mask_percentage, 
                            time_masking_max_percentage=mask_percentage,
                            use_zero=False
                            )
                        ), # Apply SpecAugment
            # transforms.Lambda(lambda x: np.apply_along_axis(spec_augment, 2, x, mask_percentage, mask_percentage, False)), 
            transforms.ToTensor(), # normalizes values to be [0,1]
            # transforms.Normalize(mean=img_stats.get("mean"), std=img_stats.get("std"))
            ]),
            valid=transforms.Compose([
            # transforms.Lambda(lambda img: extend_clip(img, crop_width)), # extend clip < 5 secs
            transforms.ToPILImage(), # covert to PIL image
            transforms.RandomCrop(size=(mel_params.get("n_mels"), crop_width),  # CentreCrop
                                pad_if_needed=True, 
                                fill=mel_params.get("min_db")
                                ), # randomly crops 5 secs
            transforms.Resize(size=(img_size, img_size), 
                            interpolation=PIL.Image.NEAREST
                            ), # resize image to be [img_size, img_size]
            transforms.Lambda(lambda x: mono_to_color(x)), # normalize to be [0-255]
            # transforms.Lambda(lambda img:
            #                 spec_augment(img, num_mask=2, 
            #                     freq_masking_max_percentage=mask_percentage, 
            #                     time_masking_max_percentage=mask_percentage,
            #                     use_zero=False
            #                     )
            #                 ), # Apply SpecAugment
            transforms.ToTensor(), # normalizes values to be [0,1]
            # transforms.Normalize(mean=img_stats.get("mean"), std=img_stats.get("std"))
            ])
        )


def extend_clip(spec: np.ndarray, crop_width):
    """If duration is < 5 secs, fill clip by repeating 
    note: experimental!
    """
    if spec.shape[1] < crop_width:
        n_reps = math.ceil((crop_width - spec.shape[1])/spec.shape[1])
        spec = np.tile(spec, n_reps)[:, :crop_width]
    return spec

def spec_augment(x: np.ndarray, num_mask=2,
                 freq_masking_max_percentage=0.15, 
                 time_masking_max_percentage=0.3,
                 use_zero=True):
    """Apply SpecAugment to each clip
    source: https://www.kaggle.com/davids1992/specaugment-quick-implementation/
    others: https://github.com/zcaceres/spec_augment, https://github.com/DemisEom/SpecAugment
    """
    # Image.fromarray(img)
    # spec = spec.copy()
    aug_spec = []
    for i in range(x.shape[-1]):
        spec = x[...,i]
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0 if use_zero else spec.mean()

            time_percentage = random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0 if use_zero else spec.mean()

        aug_spec.append(spec)
    cat = np.stack(aug_spec, axis=-1)
    return cat

def mixup():
    pass

def add_noise():
    pass


def extract_sec_labels():
    pass