from torch import nn
import torch
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate(model, loader, device, writer=None, epoch=None, plot_cm=False, cm_out_dir="eval/plots"):
    loss_fn = nn.CrossEntropyLoss()
    model.eval() 
    
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    
    with torch.no_grad(): 
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            loss = loss_fn(output, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if plot_cm:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    
    if writer and epoch is not None:
        writer.add_scalar("Loss/Validation", avg_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)

    if plot_cm:
        cm = confusion_matrix(all_labels, all_preds)
        
        epoch_str = f"_epoch_{epoch}" if epoch is not None else "_final"
        out_filename = os.path.join(cm_out_dir, f"confusion_matrix{epoch_str}.png")
        
        num_classes = len(np.unique(all_labels)) 
        plot_confusion_matrix(cm, num_classes, out_filename)
        
    return avg_loss, accuracy


def plot_confusion_matrix(cm, num_classes, out_filename):
    plt.figure(figsize=(12, 10))
    
    show_numbers = True if num_classes <= 20 else False
    
    sns.heatmap(cm, annot=show_numbers, fmt='d', cmap='Blues', 
                cbar=True, square=True)
    
    plt.title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    print(f"Confusion Matrix gespeichert unter: {out_filename}")
    plt.close()