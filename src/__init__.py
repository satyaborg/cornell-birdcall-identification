from src.config import Config
from src import models, trainer, dataset, transforms, preprocessor, utils

config = Config().get_yaml()

if config["debug"]:
    pass