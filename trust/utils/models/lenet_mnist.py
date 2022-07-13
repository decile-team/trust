'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


class LeNetMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x, last=False):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        x = self.fc3(out)
        if(last):
            return x, out
        else:
            return x

    def get_embedding_dim(self):
        return 84
