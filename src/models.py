import torch.nn as nn
from src.torchvggish import torchvggish

class SimpleConvNet(nn.Module):
    """Simple CNN; returns raw logits
    only works with 224x224 images
    """
    def __init__(self, n_classes: int):
        super(SimpleConvNet, self).__init__()
        self.n_classes = n_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=32, 
                      kernel_size=5, 
                      stride=1, 
                      bias=True
                     ), # [32, 220, 220]
            nn.MaxPool2d(kernel_size=2,
                         stride=2 # Default value: stride=kernel_size
                        ), # [32, 110, 110] 
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32) # channels
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=5, 
                      stride=1, 
                      bias=True
                     ), # [64, 106, 106]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2
                        ), # [64, 53, 53]
            nn.BatchNorm2d(num_features=64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                      out_channels=256, 
                      kernel_size=5, 
                      stride=8, 
                      bias=True
                     ), # [256, 7, 7]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5,
                         stride=2
                        ), # [256, 2, 2]
            nn.BatchNorm2d(num_features=256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256*2*2, # calculated from get_out_size()
                      out_features=1024, 
                      bias=True
                     ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, 
                      out_features=self.n_classes, 
                      bias=True
                     )
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # flatten -> [bs, 1024]
        x = self.classifier(x)
        return x

class ConvNet(nn.Module):
    """Processes 5 sec clips
    Image size: 224 x 224
    n_layers = 5
    https://cs231n.github.io/convolutional-networks/
    when 3x3 filters padding 1 
    5x5 filters padding 2
    7x7 filters padding 3
    """
    def __init__(self, n_classes: int):
        super(ConvNet, self).__init__()
        self.fcn = 1024 # 4096
        self.n_classes = n_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=32, 
                      kernel_size=7, 
                      stride=1, 
                      padding=3,
                      bias=True
                     ), # [32, 224, 224]
            nn.MaxPool2d(kernel_size=2,
                         stride=2 # Default value: stride=kernel_size
                        ), # [32, 112, 112]
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32) # channels
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=5, 
                      stride=1, 
                      padding=2,
                      bias=True
                     ), # [64, 112, 112]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2
                        ), # [64, 56, 56]
            nn.BatchNorm2d(num_features=64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=5, 
                      stride=1, 
                      padding=2,
                      bias=True
                     ), # [128, 56, 56]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2
                        ), # [128, 28, 28]
            nn.BatchNorm2d(num_features=128)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, 
                      out_channels=256, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1,
                      bias=True
                     ), # [256, 28, 28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2
                        ), # [256, 14, 14]
            nn.BatchNorm2d(num_features=256)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, 
                      out_channels=512, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1,
                      bias=True
                     ), # [512, 14, 14]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2
                        ), # [512, 7, 7]
            nn.BatchNorm2d(num_features=512)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7, # 25088
                      out_features=self.fcn, 
                      bias=True
                     ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.fcn, 
                      out_features=self.n_classes, 
                      bias=True
                     )
        )
        # Try GlobalAvgPooling (GAP)!
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.classifier(x)
        return x

class DeepConvNet(nn.Module):
    """Processes 5 sec clips
    Image size: 224 x 542
    n_layers = 5
    https://cs231n.github.io/convolutional-networks/
    when 3x3 filters padding 1 
    5x5 filters padding 2
    7x7 filters padding 3
    """
    def __init__(self, n_classes: int):
        super(DeepConvNet, self).__init__()
        self.fcn = 1024 # 4096
        self.n_classes = n_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=32, 
                      kernel_size=7, 
                      stride=1, 
                      padding=3,
                      bias=True
                     ), # [32, 224, 542]
            nn.MaxPool2d(kernel_size=2,
                         stride=2 # Default value: stride=kernel_size
                        ), # [32, 112, 271]
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32) # channels
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=5, 
                      stride=1, 
                      padding=2,
                      bias=True
                     ), # [64, 112, 271]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,3),
                         stride=(2,2)
                        ), # [64, 56, 135]
            nn.BatchNorm2d(num_features=64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=5, 
                      stride=1, 
                      padding=2,
                      bias=True
                     ), # [128, 56, 135]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,3),
                         stride=(2,2)
                        ), # [128, 28, 67]
            nn.BatchNorm2d(num_features=128)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, 
                      out_channels=256, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1,
                      bias=True
                     ), # [256, 28, 67]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,3),
                         stride=(2,2)
                        ), # [256, 14, 33]
            nn.BatchNorm2d(num_features=256)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, 
                      out_channels=512, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1,
                      bias=True
                     ), # [512, 14, 33]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,3),
                         stride=(2,2)
                        ), # [512, 7, 16]
            nn.BatchNorm2d(num_features=512)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*16, # 57344
                      out_features=self.fcn, 
                      bias=True
                     ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.fcn, 
                      out_features=self.n_classes, 
                      bias=True
                     )
        )
        # Try GlobalAvgPooling (GAP)!
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.classifier(x)
        return x

class Vggish():
    """
    Released by Google in 2017, this model extracts 128-dimensional embeddings from ~1 second long audio signals. The model was trained on a large YouTube dataset (a preliminary version of what later became YouTube-8M).

    Number of layers: 25 | Parameter count: 72,141,184 | Trained size: 289 MB |
    Training Set Information:
    Preliminary version of the YouTube-8M dataset, a large-scale labeled video dataset that consists of millions of YouTube video IDs, with high-quality machine-generated annotations from a diverse vocabulary of 3,800+ visual entities.
    """
    pass


class RCNN(nn.Module):
    """Recurrent Convolutional Network"""
    pass

class LSTM(nn.Module):
    pass