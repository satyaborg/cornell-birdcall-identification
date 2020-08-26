# #%%
# %load_ext autoreload
# %autoreload 2
# to run as a Jupyter cell in VSCode add #%%
from src import config
from src.trainer import Trainer
from src.preprocessor import Preprocessor
from src.utils import show_melspec, show_img, get_out_size, get_stats

if __name__ == "__main__":
    if config["preprocess"]:
        preprocessor = Preprocessor(**config)
        preprocessor.resample_audio()

    trainer = Trainer(**config)
    trainer.run()
    
    # testing --
    # idx = 300
    # train, test = trainer.read_data()
    # dataset = trainer.prepare_data()
    # trainloader, validloader = trainer.prepare_dataloader(dataset)
    # mean, std = get_stats(trainloader)
    # print("mean : {}, std: {}".format(mean, std))
    # show_img(dataset, idx)
