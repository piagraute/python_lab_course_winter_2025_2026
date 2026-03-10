from datetime import datetime
from typing import Literal

import torch
import torch.nn as nn
from src.data.augmentation.compose_creator import get_compose
from src.data.dataset_loader import load_dataloaders, load_dataset
from src.models.model_loader import load_model
from src.toolbox.config_loader import config_loader
from src.toolbox.logger import get_logger
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def main(
    model_str: Literal["cnn", "mobilenetv2"],
    aug: Literal["baseline", "classic", "advanced"],
    is_training: bool,
):
    experiment_name = f"{model_str}_{aug}Aug_{datetime.now().strftime('%Y%m%d_%H%M')}"
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")

    device = torch.cude("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger()
    config = config_loader(model_str)
    train_augementation = get_compose(model_str, aug)
    test_augmentation = get_compose(model_str, "baseline")
    train_set, test_set = load_dataset(
        "GTSRB",
        logger,
        train_transform=train_augementation,
        test_transform=test_augmentation,
    )
    model = load_model(model_str)
    if is_training:
        loader = load_dataloaders(
            train_set, config["batch_size"], config["num_workers"]
        )
        train(model, loader, device, config, writer)
    else:
        loader = load_dataloaders(test_set, config["batch_size"], config["num_workers"])
        #TODO


def train(model, loader, device, config, writer):
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"])

    model.train()

    for epoch in tqdm(range(config["num_epochs"]), desc="Training Epochs"):

        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward Pass
            output = model(images)
            loss = loss_fn(output, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(loader)

        writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
