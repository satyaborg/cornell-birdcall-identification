import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class BirdcallDataset(Dataset):
    """Custom pytorch dataset"""
    def __init__(self, data: pd.DataFrame, transform=None, audio_dir=None):
        self.data = data
        self.transform = transform
        self.audio_dir = audio_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        ebird_code = row.ebird_code
        fname = ebird_code + '/' + row.filename[:-4] + '.npy'
        fpath = self.audio_dir + '/' + fname
        # duration = row.duration
        label = row.label
        try:
            sample = np.load(fpath, allow_pickle=True)
            if self.transform:
                sample = self.transform(sample)
            return sample, int(label)
        except Exception as e:
            print(e)
            pass # or return a empty np.ndarray

# class ValidDataset(Dataset):
#     """Custom valid dataset"""
#     def __init__(self, data: pd.DataFrame, transform=None, audio_dir=None):
#         self.data = data
#         self.transform = transform
#         self.audio_dir = audio_dir
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx: int):
#         samples = []
#         row = self.data.iloc[idx]
#         ebird_code = row.ebird_code
#         fname = ebird_code + '/' + row.filename[:-4] + '.npy'
#         fpath = self.audio_dir + '/' + fname
#         # duration = row.duration
#         label = row.label
#         try:
#             sample = np.load(fpath, allow_pickle=True)
#             if self.transform:
#                 # get 5 transformed inputs
#                 for i in range(0,5):
#                     samples.append(self.transform(sample))
#             return np.ndarray(samples), int(label)
#         except Exception as e:
#             print(e)
#             pass # or return a empty np.ndarray
