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

def show_img(dataset, i):
    img, label = dataset[i]
    print("shape: {}, label: {}".format(img.shape, label))
    print("max: {}, min: {}".format(img.max(), img.min()))
    plt.imshow(img[0,:,:])

def show_waveplot(path, **config): 
    x = np.load(config.get("paths")["audio"] + "/" + path)
    # plt.figure(figsize=(14, 5))
    return librosa.display.waveplot(x, sr=config.get("sr"))

def play_audio(path, **config):
    pass

def show_melspec(path, **config):
    # https://stackoverflow.com/a/38923511/8277194
    # 512/32000 per hop
    x = np.load(config.get("paths")["audio"] + "/" + path)[:, :]
    print("mel-spec shape: {}".format(x.shape))
    # times = librosa.frames_to_time(x[1], sr=config.get("sr"))
    # print(times)

    librosa.display.specshow(x, y_axis='mel', x_axis='time', 
                             sr=config.get("sr"),
                             hop_length=config["mel_params"].get('hop_length', 512), 
                             fmin=config["mel_params"].get("fmin"), fmax=config["mel_params"].get("fmax")
                            # cmap=cm.jet # change color map
                            );
    # plt.figure(figsize=(25,60))
    # plt.colorbar(format='%+2.0f dB')
    # plt.title(path)
    # plt.show();

# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
def get_out_size(width_in: int, kernel_size: int, stride: int, padding: int):
    return ((width_in - kernel_size + 2 * padding)/stride) + 1

def get_padding(width_in: int, width_out: int, kernel_size: int, stride: int):
    # (kernel_size - 1)//2
    return (stride * (width_out - 1) + kernel_size - width_in) // 2

def check_npy(path):
    """check if npy files are valid"""
    try:
        np.load(path, allow_pickle=True)
    except Exception as e:
        print(e)
        pass # or return a empty np.ndarray

    