import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstCNN(nn.Module):
    def __init__(self):
        super(FirstCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_prob = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.conv_boxes = nn.Conv2d(32, 4, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool4(x)

        x_prob = torch.sigmoid(self.conv_prob(x))
        x_boxes = self.conv_boxes(x)

        gate = torch.where(x_prob > 0.5, torch.ones_like(x_prob), torch.zeros_like(x_prob))
        x_boxes = x_boxes * gate

        x_boxes = x_boxes.view(x_boxes.size(0), -1)
        x_prob = x_prob.view(x_boxes.size(0), -1)
        out = torch.cat((x_prob, x_boxes), 1)

        return out

