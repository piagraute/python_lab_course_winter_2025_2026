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
- Advanced (Baseline + AutoAugment)

### Setup
- Loss function: CrossEntropy Loss
- Optimizer: AdamW
- LR Scheduler: ReduceLROnPlateau
- Tracking: Tensorboard
- Metrics: Accuracy, Macro F1-Score

### TODO
* alle 20 epochs weights speichern
* best model 
* tensorboard
* fire
* training
* präsentation

### run training
```uv run main.py --model_str=<MODEL> --aug=<AUGMENTATION> --is_training=<BOOL>```

example:
```uv run main.py --model_str=cnn --aug=classic --is_training=True```

shorter:
```uv run main.py cnn classic True```

`model_str`specifies which model configuration should be used.\
Available models: `cnn`, `mobilenetv2`

`aug` defines which data augmentation strategy should be applied to the dataset.\
Possible values: `baseline`, `classic`, `advanced`

`is_training`determines whether the model should bei trained.\
`True`: train the model\
`False`: skip training (e.g. only evaluation)


### run tensorboard
uv run tensorboard --logdir=runs/