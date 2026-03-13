# cnn (baseline, classic, advanced)
# mobilenetv2 (baseline, classic, advanced)
from main import main
from src.toolbox.logger import get_logger
import fire


def full_run(is_training: bool = False, checkpoint_path: str = None):
    logger = get_logger()
    models = ["mobilenetv2"] #["cnn", "mobilenetv2"]
    aug_levels = ["baseline", "classic", "advanced"]
    for model in models:
        for aug_level in aug_levels:
            logger.info("")
            logger.info("="*60)
            logger.info(f"Starting Training --> Model {model} | Aug-Level {aug_level}")
            logger.info("="*60)
            logger.info("")
            main(model, aug_level, is_training, checkpoint_path)

if __name__ == "__main__":
    fire.Fire(full_run)