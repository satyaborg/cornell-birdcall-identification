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
    # idx = 3
    # train, test = trainer.read_data()
    # train.ebird_code.value_counts().plot(kind="bar")
    # datasets = trainer.prepare_data()
    # path = train.iloc[idx].ebird_code+"/"+train.iloc[idx].filename[:-4]+".npy"
    # print(path)
    # show_melspec(path, **config)
    # trainloader, validloader = trainer.prepare_dataloader(dataset)
    # mean, std = get_stats(trainloader)
    # print("mean : {}, std: {}".format(mean, std))
    # show_img(datasets["train"], idx, dim=1)
