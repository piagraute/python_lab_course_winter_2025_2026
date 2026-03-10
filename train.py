import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Literal
from src.data.dataset_loader import load_dataset
from src.toolbox.logger import get_logger


def main(model: Literal["cnn", "mobilenetv2"], aug: Literal["baseline", "classic", "advanced"]):
    experiment_name = f"{model}_{aug}Aug_{datetime.now().strftime('%Y%m%d_%H%M')}"
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
    logger = get_logger()

    train_set, test_set = load_dataset("GTSRB", logger)

    #model = load_model(model)
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