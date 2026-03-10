from src.toolbox.logger import get_logger
from src.data.dataset_loader import load_dataset, plot_dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Resize, Compose
from torchsummary import summary
from logging import Logger



class CNN(nn.Module):
    def __init__(self, logger:Logger, nclasses=43):
        super(CNN, self).__init__()
        self.logger = logger
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True)
        )
        
        # Calculate the size after convolutions for 32x32 input:
        # After conv1 (kernel=7): 32-7+1 = 26
        # After pool: 26/2 = 13
        # After conv2 (kernel=5): 13-5+1 = 9
        # After pool: 9/2 = 4
        # Result: 10 * 4 * 4 = 160
        
        # Regressor for the affine transformation matrix
        # Feature maps: 160 -> 32 -> 6 (for 2x3 affine matrix)
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        # CNN layers
        # Feature maps: 3 -> 100 -> 150 -> 250
        # Filter sizes: [5, 3, 3]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=100, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=150, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=150, out_channels=250, kernel_size=3)
        
        # Calculate size after CNN convolutions for 32x32 input:
        # After conv1 (kernel=5): 32-5+1 = 28
        # After pool: 28/2 = 14
        # After conv2 (kernel=3): 14-3+1 = 12
        # After pool: 12/2 = 6
        # After conv3 (kernel=3): 6-3+1 = 4
        # After pool: 4/2 = 2
        # Result: 250 * 2 * 2 = 1000
        
        # Fully connected layers
        # Feature maps: 1000 -> 350 -> nclasses
        self.fc1 = nn.Linear(in_features=250 * 2 * 2, out_features=350)
        self.fc2 = nn.Linear(350, nclasses)
        
    def stn(self, x):
        """Spatial Transformer Network forward pass"""
        # Localization network
        xs = self.localization(x)
        xs = torch.flatten(xs, 1)
        
        # Compute theta (affine transformation parameters)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # Generate sampling grid and sample the input
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
        return x
    
    def forward(self, x):
        # Apply spatial transformation
        x = self.stn(x)
        
        # CNN layers with ReLU and max pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        self.logger.info("CNN model with Spatial Transformer Network defined")
        
        return x
    
        
