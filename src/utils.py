import numpy as np
import random
import torch
import librosa
import matplotlib.pyplot as plt
import os

def set_random_seeds(seed: int = 42):
    random.seed(seed) # python random seed 
    np.random.seed(seed) # numpy random seeda
    os.environ["PYTHONHASHSEED"] = str(seed)# environ seeds
    torch.manual_seed(seed) # pytorch seed
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

# to clear tensors from GPU
def del_cuda_tensor(x):
    del x
    torch.cuda.empty_cache()

def plot_waveplot(x): 
    plt.figure(figsize=(14, 5))
    return librosa.display.waveplot(x, sr=sr);

def show_melspec(path):
    x = np.load(path)
    print(x.shape)
    librosa.display.specshow(x, y_axis='mel', x_axis='time', 
                             hop_length=mel_params.get('hop_length', 512), 
                             fmin=mel_params['fmin'], fmax=mel_params['fmax'],
#                              cmap=cm.jet # change color map
                            );
#     plt.figure(figsize=(25,60))

    plt.colorbar(format='%+2.0f dB')
    plt.title(path)
    plt.show();
    return x

# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
def get_out_size(width_in: int, kernel_size: int, stride: int, padding: int):
    return ((width_in - kernel_size + 2 * padding)/stride) + 1

def get_padding(width_in: int, width_out: int, kernel_size: int, stride: int):
    return (stride * (width_out - 1) + kernel_size - width_in) // 2

def check_npy(path):
    """check if npy files are valid"""
    try:
        np.load(path, allow_pickle=True)
    except Exception as e:
        print(e)
        pass # or return a empty np.ndarray

    