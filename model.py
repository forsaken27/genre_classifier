import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        ##################################### Block 1 ######################################
        # INPUT: (batch, 3, 288, 432)
        self.b1_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.b1_batchnorm1 = nn.BatchNorm2d(32)
        self.b1_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.b1_batchnorm2 = nn.BatchNorm2d(32)
        self.b1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b1_dropout = nn.Dropout(0.2)
        # OUTPUT: (batch, 32, 144, 216)

        ##################################### Block 2 ######################################
        # INPUT: (batch, 32, 144, 216)
        self.b2_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.b2_batchnorm1 = nn.BatchNorm2d(64)
        self.b2_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.b2_batchnorm2 = nn.BatchNorm2d(64)
        self.b2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b2_dropout = nn.Dropout(0.2)
        # OUTPUT: (batch, 64, 72, 108)

        ##################################### Block 3 ######################################
        # INPUT: (batch, 64, 72, 108)
        self.b3_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.b3_batchnorm1 = nn.BatchNorm2d(128)
        self.b3_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.b3_batchnorm2 = nn.BatchNorm2d(128)
        self.b3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b3_dropout = nn.Dropout(0.3)
        # OUTPUT: (batch, 128, 36, 54)

        ##################################### Block 4 ######################################
        # INPUT: (batch, 128, 36, 54)
        self.b4_conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.b4_batchnorm1 = nn.BatchNorm2d(256)
        self.b4_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.b4_batchnorm2 = nn.BatchNorm2d(256)
        self.b4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b4_dropout = nn.Dropout(0.4)
        # OUTPUT: (batch, 256, 18, 27)

        ##################################### Classifier ######################################
        # Global Average Pooling: (batch, 256, 18, 27) -> (batch, 256, 1, 1) -> (batch, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        # OUTPUT: (batch, 10)

    def forward(self, x):
        # Block 1
        x = self.b1_conv1(x)
        x = self.b1_batchnorm1(x)
        x = F.relu(x)
        x = self.b1_conv2(x)
        x = self.b1_batchnorm2(x)
        x = F.relu(x)
        x = self.b1_pool(x)
        x = self.b1_dropout(x)

        # Block 2
        x = self.b2_conv1(x)
        x = self.b2_batchnorm1(x)
        x = F.relu(x)
        x = self.b2_conv2(x)
        x = self.b2_batchnorm2(x)
        x = F.relu(x)
        x = self.b2_pool(x)
        x = self.b2_dropout(x)

        # Block 3
        x = self.b3_conv1(x)
        x = self.b3_batchnorm1(x)
        x = F.relu(x)
        x = self.b3_conv2(x)
        x = self.b3_batchnorm2(x)
        x = F.relu(x)
        x = self.b3_pool(x)
        x = self.b3_dropout(x)

        # Block 4
        x = self.b4_conv1(x)
        x = self.b4_batchnorm1(x)
        x = F.relu(x)
        x = self.b4_conv2(x)
        x = self.b4_batchnorm2(x)
        x = F.relu(x)
        x = self.b4_pool(x)
        x = self.b4_dropout(x)

        # Classifier with Global Average Pooling
        x = self.global_pool(x)  # (batch, 256, 18, 27) -> (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256, 1, 1) -> (batch, 256)
        x = self.dropout(x)
        x = self.fc(x)

        return x
