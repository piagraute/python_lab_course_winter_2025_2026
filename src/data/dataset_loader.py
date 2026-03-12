from logging import Logger
from pathlib import Path
from typing import Literal
from src.toolbox.logger import get_logger
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor


def load_dataset(
    dataset: Literal["GTSRB"],
    logger: Logger,
    root_path: Path = Path("data"),
    train_transform: Compose = ToTensor(),
    test_transform: Compose = ToTensor(),
):  
    
    if dataset == "GTSRB":
        if Path.is_dir(root_path / "gtsrb"):
            logger.info("Found downloaded model. Skipping download...")
            is_download = False
        else: 
            is_download = True
            logger.info(f"Start loading dataset {dataset}...")
        
        train_data = datasets.GTSRB(
            root=root_path,
            split="train",
            transform=train_transform,
            download=is_download,
        )
        test_data = datasets.GTSRB(
            root=root_path,
            split="test",
            transform=test_transform,
            download=is_download,
        )
    logger.info(f"Shape of training data: {len(train_data)}")
    logger.info(f"Shape of test data: {len(test_data)}")
    return train_data, test_data

def load_dataloaders(dataset, batch_size, num_workers, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def plot_dataset(dataset, logger):
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        if isinstance(img, torch.Tensor):
            img_to_plot = img.permute(1, 2, 0).numpy()
        else:
            img_to_plot = img
        plt.imshow(img_to_plot)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logger = get_logger()
    train, test = load_dataset("GTSRB", logger)
    plot_dataset(train, logger)
    
