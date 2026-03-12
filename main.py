from datetime import datetime
from typing import Literal

import fire
import torch
from src.data.augmentation.compose_creator import create_augmentation
from src.data.dataset_loader import load_dataloaders, load_dataset
from src.experiments.eval import evaluate
from src.experiments.train import train, train_mobilenet_finetuning
from src.models.model_loader import load_model
from src.toolbox.config_loader import config_loader
from src.toolbox.logger import get_logger
from src.toolbox.seed import set_seed
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def main(
    model_str: Literal["cnn", "mobilenetv2"] = "cnn",
    aug: Literal["baseline", "classic", "advanced"] = "advanced",
    is_training: bool = True,
    checkpoint_path: str = None
):
    experiment_name = f"{model_str}_{aug}Aug_{datetime.now().strftime('%Y%m%d_%H%M')}"
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
  
    logger = get_logger()

    logger.info(f"Experiment start | model = {model_str} | augmentation = {aug} | training = {is_training}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(3)
    logger.info("Use 3 CPU threads for training")
    if device.type == "cuda":
        torch.cuda.memory.set_per_process_memory_fraction(0.33) #set to 0.16 if 2 training runs are executed in parallel
        logger.info("Using 33 percent of GPU")
   

 

    config = config_loader(model_str)
    set_seed(config["seed"])

    train_augmentation = create_augmentation(model_str, aug)
    test_augmentation = create_augmentation(model_str, "baseline")

    logger.info(f"Load dataset for training with augmentation ({aug})...")
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
    model = model.to(device)

    start_epoch = 0
    best_val_acc = 0.0
    optimizer_state = None

    if checkpoint_path is not None:
        if Path(checkpoint_path).is_file():
            logger.info(f"Load checkpoint from: {checkpoint_path}")

            # load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if is_training:
                optimizer_state = checkpoint.get('optimizer_state_dict')
                start_epoch = checkpoint.get('epoch', 0)
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                logger.info(f"Training continued at epoch {start_epoch + 1}.")
            else:
                logger.info("Modell loaded for inference.")
        else:
            logger.error(f"Couldn't find checkpoint at: {checkpoint_path}")
            return 

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
            real_train_set, config["batch_size"], config["num_workers"], shuffle=True
        )
        val_loader = load_dataloaders(
            val_set, config["batch_size"], config["num_workers"], shuffle=False
        )

        logger.info("Start training...")
        if model_str == "mobilenetv2":
            train_mobilenet_finetuning(
                model, 
                train_loader, 
                val_loader, 
                device, 
                config, 
                writer, 
                logger
            )
        else:
            train(
                model, 
                train_loader, 
                val_loader, 
                device, 
                config, 
                writer, 
                logger,
                optimizer_state=optimizer_state,
                start_epoch=start_epoch,
                best_val_acc=best_val_acc
            )

    else:
        # evaluation
        logger.info("Start evaluation on test set...")
        test_loader = load_dataloaders(
            test_set, config["batch_size"], config["num_workers"], shuffle=False
        )

        test_loss, test_acc = evaluate(model, test_loader, device, plot_cm=True)
        logger.info(
            f"Final test scores --> Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}%"
        )


if __name__ == "__main__":
    fire.Fire(main)