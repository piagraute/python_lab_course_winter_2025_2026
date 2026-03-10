from typing import Literal 
from src.data.augmentation.baseline import create_baseline_augmentation
from src.data.augmentation.classic import create_classic_augmentation
from src.data.augmentation.advanced import create_advanced_augmentation
from torchvision.transforms import Compose, Resize, ToTensor

def get_compose(model: Literal["cnn", "mobilnetv2"], compose_level: Literal["baseline", "classic", "advanced"]):
    if compose_level == "baseline":
        return create_baseline_augmentation(model)
    if compose_level == "classic":
        return create_classic_augmentation(model)
    if compose_level == "advanced":
        return create_advanced_augmentation(model)
    
