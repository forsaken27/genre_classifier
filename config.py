import torch.nn as nn
import torch

class Config:
    def __init__(self):
        #############################################
        #          Configurations for CNN           #
        #############################################
        self.batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        self.learning_rate = 0.002
        self.epochs = 150

