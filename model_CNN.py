import torch.nn as nn
import torch.nn.functional as F

num_classes = 3

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16*53*53, 120)
        self.fc2 = nn.Linear(120, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = F.adaptive_avg_pool2d(x, (53, 53))

        x = x.view(-1, 16*53*53)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x