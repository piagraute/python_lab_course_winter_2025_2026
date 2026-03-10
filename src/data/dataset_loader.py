from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Literal
import torch
import matplotlib.pyplot as plt
from logging import Logger

def load_dataset(dataset: Literal["GTSRB"], logger: Logger):
    if dataset == "GTSRB":
        train_data = datasets.GTSRB(
            root = 'data',
            train = True,
            transform = ToTensor(),
            download = True,
        )
        test_data = datasets.MNIST(
            root = 'data',
            train = False,
            transform = ToTensor()
        )
    logger.info(f"Shape of training data: {train_data.data.size()}")
    logger.info(f"Shape of test data: {test_data.data.size()}")
    

def plot_dataset(dataset, logger):
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()