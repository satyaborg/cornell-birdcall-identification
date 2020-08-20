from src import config
from src.trainer import Trainer
from src.preprocessor import Preprocessor

if __name__ == "__main__":

    if config["preprocess"]:
        preprocessor = Preprocessor(**config)
        preprocessor.resample_audio()

    trainer = Trainer(**config)
    trainer.train()