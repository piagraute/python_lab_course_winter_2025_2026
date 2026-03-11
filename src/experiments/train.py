
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from src.experiments.eval import evaluate
from pathlib import Path
import torch

def train(
        model, 
        train_loader, 
        val_loader, 
        device, 
        config, 
        writer, 
        logger,
        optimizer_state=None,
        start_epoch=0,
        best_val_acc=0.0
    ):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        
    save_dir = Path(writer.log_dir)

    for epoch in tqdm(range(start_epoch, config["num_epochs"]), 
                      desc="Training Epochs", 
                      initial=start_epoch, 
                      total=config["num_epochs"]):
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

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        checkpoint = {
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config
        }

        save_freq = config.get("epoch_save_freq", 5)
        if epoch % save_freq == 0:
            torch.save(checkpoint, save_dir / f"checkpoint_{epoch+1}.pt")
            logger.info(f"New checkpoint {epoch} saved.")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, save_dir / "best_model.pt")
            logger.info(f"New best model saved (Val Acc: {best_val_acc})")