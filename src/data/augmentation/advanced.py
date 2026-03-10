from torchvision.transforms import Compose
from typing import Literal
from src.data.augmentation.classic import create_classic_augmentation


def create_advanced_augmentation(model: Literal["cnn", "mobilenetv2"]) -> Compose:
    classic_aug = create_classic_augmentation(model)
    ...
    #TODO
    
