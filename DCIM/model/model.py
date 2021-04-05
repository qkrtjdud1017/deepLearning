import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class DCIMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))

        self.fc1 = nn.Linear(in_features=256*8*8, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)

        self.out = nn.Linear(in_features=4096, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.dropout(self.fc1(x), p=0.5)
        x = F.dropout(self.fc2(x), p=0.5)

        x = self.out(x)
        return x