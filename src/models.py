import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

class SimpleConvNet(nn.Module):
    """Simple CNN; returns raw logits"""
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

class Vggish():
    pass

class RCNN(nn.Module):
    """Recurrent Convolutional Network"""
    pass

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = 3
        self.padding = self.kernel_size//2
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, self.kernel_size, padding=self.padding)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        device = 'cuda' # Gotta bubble this parameter upwards later. Keeping it here for intial debugging
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)).to(device),
                Variable(torch.zeros(state_size)).to(device)
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        #return hidden, cell
        return hidden

# TO_DO: Include the loop with the sequence length...