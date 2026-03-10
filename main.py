from datetime import datetime
from typing import Literal

import torch
from src.data.augmentation.compose_creator import create_augmentation
from src.data.dataset_loader import load_dataloaders, load_dataset
from src.experiments.eval import evaluate
from src.experiments.train import train
from src.models.model_loader import load_model
from src.toolbox.config_loader import config_loader
from src.toolbox.logger import get_logger
from src.toolbox.seed import set_seed
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter


def main(
    model_str: Literal["cnn", "mobilenetv2"],
    aug: Literal["baseline", "classic", "advanced"],
    is_training: bool,
):
    experiment_name = f"{model_str}_{aug}Aug_{datetime.now().strftime('%Y%m%d_%H%M')}"
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")


    torch.set_num_threads(3)
    logger.info("Use 3 CPU threads for training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger()

    config = config_loader(model_str)
    set_seed(config["seed"])

    train_augmentation = create_augmentation(model_str, aug)
    test_augmentation = create_augmentation(model_str, "baseline")

    logger.info("Load dataset for training...")
    train_set_aug, test_set = load_dataset(
        "GTSRB",
        logger,
        train_transform=train_augmentation,
        test_transform=test_augmentation,
    )

    logger.info("Load dataset for validation without augmentation...")
    train_set_clean, _ = load_dataset(
        "GTSRB",
        logger,
        train_transform=test_augmentation,
        test_transform=test_augmentation,
    )

    model = load_model(model_str, logger)

    if is_training:
        total_train_size = len(train_set_aug)
        train_size = int(0.8 * total_train_size)

        logger.info("Shuffle dataset...")
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_train_size, generator=generator).tolist()

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        logger.info("Create subsets for train and validation set...")
        real_train_set = Subset(train_set_aug, train_indices)
        val_set = Subset(
            train_set_clean, val_indices
        )  # <- Hier ist der Zauber! Keine Augmentations!

        logger.info("Load data loaders...")
        train_loader = load_dataloaders(
            real_train_set, config["batch_size"], config["num_workers"]
        )
        val_loader = load_dataloaders(
            val_set, config["batch_size"], config["num_workers"]
        )

        logger.info("Start training...")
        train(model, train_loader, val_loader, device, config, writer, logger)

    else:
        # evaluation
        logger.info("Start evaluation on test set...")
        test_loader = load_dataloaders(
            test_set, config["batch_size"], config["num_workers"]
        )

        test_loss, test_acc = evaluate(model, test_loader, device)
        logger.info(
            f"Final test scores --> Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}%"
        )


if __name__ == "__main__":
    main("cnn", "advanced", True)
