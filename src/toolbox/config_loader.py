from typing import Literal 
from typing import Dict, Any, Literal
import yaml

def config_loader(model_str: Literal["cnn", "mobilenetv2"], config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"Config file '{config_path}' is empty or not properly formatted.")
    
    if model_str not in config:
        raise ValueError(f"Model '{model_str}' not found in config file.")
    
    return config[model_str]