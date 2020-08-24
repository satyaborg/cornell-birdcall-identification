# general
import os, sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import torch
from torch.functional import F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from src.models import SimpleConvNet # RCNN, LSTM
from src.dataset import TrainDataset
from src.transforms import transformations 
from src.utils import set_random_seeds
from fastprogress import progress_bar
from time import strftime
import logging
import optuna

# logs to file
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', 
                    filename="logs/" + strftime("%Y_%m_%d_%H_%M.log"), 
                    filemode="w", 
                    level=logging.INFO
                    )
logger = logging.getLogger("train_log")

# logs to console
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Trainer(object):
    def __init__(self, **config):
        self.best_f1 = 0.
        self.start_epoch = 0
        self.with_cv = config["with_cv"]
        self.ebird_code = {}
        self.device = config["device"]
        self.paths = config["paths"]
        self.img_size = config["img_size"]
        self.n_classes = config.get("n_classes")
        self.threshold = config.get("threshold")
        self.sr = config.get("sr", 32000)
        self.hyperparams = config["hyperparams"]
        self.mel_params = config["mel_params"]
        self.epochs = config["hyperparams"].get("epochs", 100)
        self.writer = SummaryWriter()
        self.model = SimpleConvNet(n_classes=self.n_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams.get("lr"))
        self.dataset = self.prepare_data()
        

    def read_data(self):
        train = pd.read_csv(self.paths.get("train", "data/train.csv"))
        # special cases:
        train = train.loc[(train.filename != 'XC195038.mp3') & (train.filename != 'XC555482.mp3')]
        with open(self.paths.get("ebird_code"), "r") as f:
            self.ebird_code = json.load(f)
        train['label'] = train.ebird_code.apply(lambda x : self.ebird_code[x])
        test = pd.read_csv(self.paths.get("test", "data/test.csv"))
        return train, test

    def prepare_data(self):
        df, _ = self.read_data()
        # get train data and apply transformations 
        transform = transformations(self.img_size)
        dataset = TrainDataset(df, self.sr, self.mel_params, transform, self.paths.get("audio"))
        return dataset
    
    def load_weights(self, model, optimizer, scheduler):
        # check for existing model.pt and load the same
        checkpoint = torch.load(self.paths.get("model_dir"), map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] # to resume training from this epoch
        self.best_f1 = checkpoint['f1_score']
        logger.info('==> checkpoint found .. start_epoch: {}, best_f1: {:.6f}'.format(self.start_epoch, self.best_f1))
        return model, optimizer, scheduler

    def train(self, dataset, load_saved, **kwargs):
        # run this to delete model and reclaim memory
        if "model" in locals(): 
            del model
            gc.collect()
            logger.info("==> Model deleted")
        
        avg_train_loss = []
        avg_valid_loss = []
        # obtain training indices that will be used for validation
        if self.with_cv:
            train_idx, valid_idx = kwargs.get("splits")
        else:
            num_train = len(dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            split = int(np.floor(self.hyperparams.get("valid_size") * num_train))
            train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        trainloader = DataLoader(dataset,
                                batch_size=self.hyperparams.get("batch_size"),
                                sampler=train_sampler,
                                num_workers=self.hyperparams.get("num_workers"),
                                pin_memory=True,
                                # collate_fn=collate_fn,
                                drop_last=True
                                )
        validloader = DataLoader(dataset, 
                                batch_size=self.hyperparams.get("batch_size"), 
                                sampler=valid_sampler, 
                                num_workers=self.hyperparams.get("num_workers"), 
                                pin_memory=True,
                                # collate_fn=collate_fn,
                                drop_last=True
                                )
        
        # intialize model, criterion, optimizer, scheduler
        model = self.model
        model.to(self.device)
        logger.info("==> model initialized ..")
        # note: always use BCEWithLogitsLoss instead of (F.sigmoid + BCELoss) for numerical stability
        criterion = torch.nn.BCEWithLogitsLoss()
        if self.hyperparams.get("scheduler") == "cosineannealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                T_max=self.hyperparams.get("t_max")
            )
        # not to be used when doing cv
        if not self.with_cv and os.path.exists(self.paths.get("model_dir")) and load_saved:
            model, optimizer, scheduler = self.load_weights(model, self.optimizer, scheduler)
        else:
            logger.info('==> Training') if not load_saved else logger.info('==> No checkpoints found / training from scratch ..')
            optimizer = self.optimizer
        
        logger.info('==> Training started ..')
        for epoch in range(self.start_epoch, self.epochs):
            logger.info('**********************\n')
            # keep track of training and validation loss
            train_loss, valid_loss = 0., 0.
            train_epoch_loss, valid_epoch_loss = [], []
            # training mode
            model.train()
            scheduler.step()
            logger.info('==> Current LR: {}'.format(scheduler.get_lr()))
            for step, (images, labels) in enumerate(progress_bar(trainloader)):
                # one-hot encode labels
                labels = labels.unsqueeze(-1)
                targets = torch.zeros(labels.size(0), model.n_classes).scatter_(1, labels, 1.)
                inputs, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad() # clear gradients
                outputs = model(inputs) # forward pass
                loss = criterion(outputs, targets) # compute loss
                loss.backward() # backward pass 
                optimizer.step() # weight update
                train_loss += loss.item()
                train_epoch_loss.append(loss.item())
                self.writer.add_scalar('Train loss', loss.item(), step + 1)
                # ===================log========================
                if (step + 1) % 50 == 0:
                    logger.info(
                        "step: [{}/{}], epochs: [{}/{}], train loss: {:.4f}".format(
                            step + 1,
                            len(trainloader),
                            epoch + 1,
                            self.epochs,
                            train_loss/50,
                        )
                    )
                    train_loss = 0.
                # ==============================================

            avg_train_loss.append(np.mean(train_epoch_loss))
            logger.info('==> Validation ..')
            # validation
            with torch.no_grad(): # turn off gradient calc
                model.eval() # evaluation mode
                y_true, y_pred = [], []
                for step, (images, labels) in enumerate(progress_bar(validloader)):
                    # one-hot encode labels
                    labels = labels.unsqueeze(-1)
                    targets = torch.zeros(labels.size(0), model.n_classes).scatter_(1, labels, 1.)
                    inputs, targets = images.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    preds = F.sigmoid(outputs) # get the sigmoid from raw logits
                    # pred = torch.argmax(output, dim=1)
                    y_true.extend(targets.cpu().detach().numpy().tolist())
                    y_pred.extend(preds.cpu().detach().numpy().tolist())
                    valid_loss += loss.item()
                    valid_epoch_loss.append(loss.item())
                    self.writer.add_scalar('Valid loss', loss.item(), step + 1)
                    # ===================log========================
                    if (step + 1) % 50 == 0:
                        logger.info(
                            "step: [{}/{}], epochs: [{}/{}], validation loss: {:.4f}".format(
                                step + 1,
                                len(validloader),
                                epoch + 1,
                                self.epochs,
                                valid_loss/50,
                            )
                        )
                        valid_loss = 0.
                    # ==============================================
            
            # metric: row-wise micro averaged F1 score
            # https://www.kaggle.com/shonenkov/competition-metrics
            y_pred = np.asarray(y_pred, dtype=np.float32)
            y_true = np.asarray(y_true, dtype=np.float32)
            micro_avg_f1 = f1_score(y_true, np.where(y_pred > self.threshold, 1, 0), average='samples') # NOT "micro"?
            scheduler.step() # call/update the scheduler
            print('epoch: [{}/{}], validation F1-score: {:.6f}'.format(epoch+1, self.epochs, micro_avg_f1))
            avg_valid_loss.append(np.mean(valid_epoch_loss)) # update average validation loss
            # save model if validation loss has decreased
            if micro_avg_f1 >= self.best_f1:
                self.writer.add_scalar('Valid F1', micro_avg_f1, epoch)
                logger.info('=> Validation F1 has increased ({:.6f} --> {:.6f})\nSaving model..'.format(
                    self.best_f1,
                    micro_avg_f1
                    )
                )
                self.best_f1 = micro_avg_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'scheduler_state_dict' : scheduler.state_dict(),
                    'train_loss' : avg_train_loss[-1],
                    'valid_loss': avg_valid_loss[-1],
                    'f1_score' : self.best_f1,
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, self.paths.get("model_dir")
                )
                logger.info('=> Model saved')
            
            logger.info('**********************\n')

            return avg_valid_loss[-1] # Optuna optimzes for this value; Minimizes this.

    def set_config(self, optimizer):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams.get("lr"))
        elif optimizer == 'MomentumSGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hyperparams.get("lr"))

    def objective(self, trial):
        load_saved = False
        optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])
        #dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)
        self.set_config(optimizer)
        valid_loss = self.train(self.dataset, load_saved)
        return valid_loss
        
    def run(self):
        """Main training function
        1. Split to K folds (stratified)
        2. Train model for max epochs from scratch for each fold
        3. Save the best F1 score model
        """
        set_random_seeds()
        load_saved = True # Load the saved checkpoint, if present
        if self.with_cv:
            logger.info("==> training w/ Kfold CV ..")
            skfold = StratifiedKFold(n_splits=self.hyperparams["cv"].get("splits"), shuffle=True, random_state=42)
            # note: for X we can simply pass a tensor of zeros 
            # source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold.split
            for fold, (train_idx, valid_idx) in enumerate(skfold.split(torch.zeros(len(self.dataset)) , self.dataset.data.label)):
                logger.info("Fold : [{}/{}]".format(fold + 1, self.hyperparams["cv"].get("splits")))
                kwargs = {"splits" : (train_idx, valid_idx)}
                self.train(self.dataset, load_saved, **kwargs)
        else:
            logger.info("==> training w/o Kfold CV ..")
            study = optuna.create_study()
            study.optimize(self.objective, n_trials=5)
            print("Optimal hyperparams: ", study.best_params)
            logger.info('=> Optimal hyperparams: {}'.format(study.best_params))
