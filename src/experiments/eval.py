from torch import nn
import torch

def evaluate(model, loader, device, writer=None, epoch=None):
    loss_fn = nn.CrossEntropyLoss()
    model.eval() 
    
    running_loss = 0.0
    correct = 0
    total = 0
    
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
            
    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    
    if writer and epoch is not None:
        writer.add_scalar("Loss/Validation", avg_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)
        
    return avg_loss, accuracy