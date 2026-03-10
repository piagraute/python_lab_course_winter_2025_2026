# PyTorch implementation of GTSRB Classificaiton

## Tasks
- Explore your dataset
- Understand your model architecture
- Train, analyze and evaluate your model
- Try to improve your results

## Dataset Infos
- Unbalanced Dataset
- 43 classes
- 50.000 images

## Approaches
### Models
- CNN (https://github.com/poojahira/gtsrb-pytorch)
- MobileNet V2 

### Augmentation
- Baseline (Resize, Normalize)
- Classic (Baseline + ColorJitter, RandomRotation (+-15), RandomAffine)
- Advanced (Classic + AutoAugment)

### Setup
- Loss function: CrossEntropy Loss
- Optimizer: AdamW
- LR Scheduler: ReduceLROnPlateau
- Tracking: Tensorboard
- Metrics: Accuracy, Macro F1-Score
