import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """Custom pytorch dataset"""
    def __init__(self, data: pd.DataFrame, sr: int, 
                mel_params: dict={}, transform=None,
                audio_dir=None):
        self.data = data
        self.sr = sr
        self.mel_params = mel_params
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