
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ColorJitter, RandomRotation, RandomAffine, AutoAugment, AutoAugmentPolicy
from typing import Literal

def get_model_specs(model: Literal["cnn", "mobilenetv2"]):
    if model == "cnn":
        return (32, 32), [0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629]
    elif model == "mobilenetv2":
        return (224, 224), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    raise ValueError(f"Unknown model: {model}")

def create_augmentation(
    model: Literal["cnn", "mobilenetv2"], 
    level: Literal["baseline", "classic", "advanced"]
) -> Compose:
    
    size, mean, std = get_model_specs(model)

    transforms_list = [Resize(size)]

    if level in ["classic", "advanced"]:
        transforms_list.extend([
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            RandomRotation(degrees=15),
            RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])

    if level == "advanced":
        transforms_list.append(AutoAugment(policy=AutoAugmentPolicy.CIFAR10))

    transforms_list.extend([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return Compose(transforms_list)