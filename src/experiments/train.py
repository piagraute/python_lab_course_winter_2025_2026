
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from src.experiments.eval import evaluate


def train(model, train_loader, val_loader, device, config, writer, logger):
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"])
    for epoch in tqdm(range(config["num_epochs"]), desc="Training Epochs"):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
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

        epoch_train_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/Train", epoch_train_loss, epoch)

        val_loss, val_acc = evaluate(model, val_loader, device, writer, epoch)

        logger.info(f"\nEpoche {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")