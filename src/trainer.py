import os
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

# modules
from src.models import SimpleConvNet # RCNN, LSTM
from src.dataset import TrainDataset
from src.transforms import transformations 
from src.utils import set_random_seeds
from src.utils import check_npy

from fastprogress import progress_bar
import logging

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("train_log")

class Trainer(object):
    def __init__(self, **config):
        self.best_f1 = 0.
        self.start_epoch = 0
        self.running_loss = 0.
        self.avg_train_loss = []
        self.avg_valid_loss = []
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
        self.writer = SummaryWriter(config["paths"].get("logs"))

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

    def train(self):
        set_random_seeds()
        # ===================model, criterion, optimizer, scheduler========================
        model = SimpleConvNet(n_classes=self.n_classes)
        model.to(self.device)
        # note: always use BCEWithLogitsLoss instead of (F.sigmoid + BCELoss) for numerical stability
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.hyperparams.get("lr"))
        if self.hyperparams.get("scheduler") == "cosineannealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                T_max=self.hyperparams.get("t_max")
            )
        # =================================================================================

        # check for existing model.pt and load the same
        if os.path.exists(self.paths.get("model_dir")):
            logging.info('==> Found checkpoint.pt ..')
            checkpoint = torch.load(self.paths.get("model_dir"), map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] # to resume training from this epoch
            self.best_f1 = checkpoint['f1_score']
            logging.info('==> Model loaded successfully')

        logger.info('==> Training started ..')
        for epoch in range(self.start_epoch, self.epochs):
            # keep track of training and validation loss
            train_loss, valid_loss = 0., 0.
            train_epoch_loss, valid_epoch_loss = [], []
            # training mode
            model.train()
            scheduler.step()
            logger.info('\n########################\n')
            logger.info('==> current LR: {}'.format(scheduler.get_lr()))
            dataset = self.prepare_data()
            skfold = StratifiedKFold(n_splits=self.hyperparams["cv"].get("splits"), shuffle=True, random_state=42)
            # note: for X we can simply pass a tensor of zeros 
            # source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold.split
            for fold, (train_idx, valid_idx) in enumerate(skfold.split(torch.zeros(len(dataset)) , dataset.data.label)):
                logger.info("Fold : [{}/{}]".format(fold, self.hyperparams["cv"].get("splits")))
                # samplers for obtaining training and validation batches
                # train_sampler = SubsetRandomSampler(train_idx)
                # valid_sampler = SubsetRandomSampler(valid_idx)
                trainset = torch.utils.data.Subset(dataset, train_idx)
                validset = torch.utils.data.Subset(dataset, valid_idx)
                # load training data in batches
                trainloader = DataLoader(trainset,
                                        batch_size=self.hyperparams.get("batch_size"),
                                        # sampler=train_sampler,
                                        num_workers=self.hyperparams.get("num_workers"),
                                        pin_memory=True,
                                        # collate_fn=collate_fn,
                                        drop_last=True
                                        )
                # load validation data in batches
                validloader = DataLoader(validset, 
                                        batch_size=self.hyperparams.get("batch_size"), 
                                        # sampler=valid_sampler, 
                                        num_workers=self.hyperparams.get("num_workers"), 
                                        pin_memory=True,
                                        # collate_fn=collate_fn,
                                        drop_last=True
                                        )

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
                            "step: [{}/{}], epoch: [{}/{}], train loss: {:.6f}".format(
                                step + 1,
                                len(trainloader),
                                epoch + 1,
                                self.epochs,
                                train_loss/50,
                            )
                        )
                        train_loss = 0.
                    # ==============================================

                self.avg_train_loss.append(np.mean(train_epoch_loss))
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
                                "step: [{}/{}], epoch: [{}/{}], validation loss: {:.6f}".format(
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
                self.avg_valid_loss.append(np.mean(valid_epoch_loss)) # update average validation loss
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
                        'best_train_loss' : self.avg_train_loss[-1],
                        'best_valid_loss': self.avg_valid_loss[-1],
                        'f1_score' : self.best_f1,
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, self.paths.get("model_dir")
                    )
                    logger.info('=> Model saved')