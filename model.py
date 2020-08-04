import torch.nn as nn
import torch
from torch import autograd


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5), 1, 0)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 120, (5, 5), 1, 0)
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.linear2 = nn.Linear(in_features=84, out_features=10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = x.view(-1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x.unsqueeze(0)

if __name__ == "__main__":
    model = LeNet()
    print(model)
    input = torch.randn((1, 1, 32, 32))
    out = model(input)
    print(out.shape)