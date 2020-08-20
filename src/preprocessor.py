# basic
import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

# multiprocessing
from joblib import Parallel, delayed, cpu_count

# audio processing
import librosa
import librosa.display
# fast chirplet transformation
# from fastchirplet import chirplet as ch

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# supresses: warnings.warn('PySoundFile failed. Trying audioread instead.') when trying to read mp3 files

class Preprocessor(object):
    def __init__(self, **config):
        self.sr = config.get("sr")
        self.verbose = config.get("verbose")

    def save_melspec(self, inpath: str, outpath: str, sr: int, **mel_params):
        try:
            x, _ = librosa.load(
                            inpath,
                            sr=self.sr, 
                            mono=True,
                            offset=0.0,
                            # duration=duration, if left out, uses complete duration of clip
                            res_type='kaiser_fast' # use kaiser fast as per competition host
            )
            x, _ = librosa.effects.trim(x)
            melspec = librosa.feature.melspectrogram(x, sr=sr, **mel_params)
            melspec = librosa.power_to_db(melspec, ref=np.max).astype(np.float32)
            # save the mel-spectogram
            np.save(outpath, melspec)
        except Exception as e:
            print(e)
            pass

    def resample_audio(self):
        # e.g. python preprocessor.py ext_data train_audio resampled_audio_ext_nz
        # python preprocessor.py /media/satya/Seagate/kaggle/cornell-birdcall-identification/data train_audio_a_m /home/satya/Projects/kaggle/cornell-birdcall-identification/data/resampled_audio_ext_a_m
        args = sys.argv
        # print(args)
        if len(args) < 4:
            print('Not enough args: 3 required')
            sys.exit()

        ROOT_DIR = Path(args[1])
        IN_DIR = ROOT_DIR/args[2]
        OUT_DIR = Path(args[3])
        train = pd.read_csv(ROOT_DIR/'train.csv')

        cols = ['ebird_code', 'duration', 'sampling_rate', 'filename', 'species', 'secondary_labels']
        df = train[cols].copy()
        df['path'] = ROOT_DIR / (args[2] + '/' + df.ebird_code + '/' + df.filename)
        ebird_code = df.ebird_code.unique().tolist()
        EBIRD_CODE = {k : str(i) for i, k in enumerate(ebird_code)}
        df['label'] = df.ebird_code.apply(lambda x : EBIRD_CODE[x])
        n_jobs = cpu_count()
        start = time.time()
        for directory in (IN_DIR).iterdir():
            ebird_code = directory.name
            fnames = os.listdir(directory)
            img_dir = OUT_DIR/ebird_code
            # check if output dir exists
            if not img_dir.exists(): 
                os.mkdir(img_dir)
            else:
                # skip
                continue
            paths = [ (os.path.join(IN_DIR/ebird_code, fname), os.path.join(img_dir, '{}.npy'.format(fname[:-4]))) for fname in fnames ]
            jobs = [ delayed(self.save_melspec)(i, o, SR, **self.mel_params) for i,o in paths ]
            out = Parallel(n_jobs=n_jobs, verbose=self.verbose)(jobs)
            print('=> ebird_code : %s processed..'%ebird_code)
        print('=> Completed in {} secs'.format(time.time() - start))