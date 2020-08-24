##%%
# %load_ext autoreload
# %autoreload 2
# to run as a Jupyter cell in VSCode add #%%
from src import config
from src.trainer import Trainer
from src.preprocessor import Preprocessor
from src.utils import show_melspec, show_img, get_out_size

if __name__ == "__main__":
    if config["preprocess"]:
        preprocessor = Preprocessor(**config)
        preprocessor.resample_audio()

    trainer = Trainer(**config)
    trainer.run()

    # df, _ = trainer.read_data()
    # mp3_paths = (df.ebird_code + '/' + df.filename).tolist()
    # paths = (df.ebird_code + '/' + df.filename.apply(lambda x : x[:-4]+".npy")).tolist()
    # %time
    # idx = 1
    # print(df.iloc[idx].duration)
    # dataset = trainer.prepare_data()
    # show_img(dataset, idx)
    # print(df.iloc[idx].duration, df.iloc[idx].filename)
    # show_melspec(paths[idx], **config)