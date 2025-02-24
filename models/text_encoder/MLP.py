import torch
import torch.nn as nn

class LinearBNReLU(nn.Module):
    def __init__(self, inchannels, outchannels) -> None:
        super(LinearBNReLU, self).__init__()
        self.linear = nn.Linear(inchannels, outchannels)
        self.bn = nn.BatchNorm1d(outchannels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MLP(nn.Module):
    def __init__(self, args, inchannels, outchannels, num_layers=2) -> None:
        super(MLP, self).__init__()
        self.args = args
        self.layer1 = LinearBNReLU(inchannels, outchannels)
        self.layers = nn.ModuleList([LinearBNReLU(outchannels, outchannels) for i in range(num_layers-1)])
        
    def forward(self,x):
        x = self.layer1(x)
        for layer in self.layers:
            x = layer(x)
        x = x.unsqueeze(1)
        return x