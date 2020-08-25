# #%%
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
    
    # test --
    # idx = 0
    # dataset = trainer.prepare_data()
    # show_img(dataset, idx)