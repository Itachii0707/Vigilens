"""Training CLI script for Veyraxis Sentinel."""

import argparse
import sys
from pathlib import Path

from veyraxis.training.trainer import VeyraxisTrainer
from veyraxis.utils.config import SentinelConfig, load_yaml_config
from veyraxis.utils.logger import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger("scripts.train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Veyraxis Sentinel object detection model.")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to training config YAML")
    parser.add_argument("--data", type=str, default=None, help="Override path to dataset.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override base model weight name (e.g. yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override execution device ('0', 'cpu', 'auto')")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    cfg_dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        cfg_dict = load_yaml_config(config_path)

    config = SentinelConfig(**cfg_dict)

    # CLI Overrides
    if args.model:
        config.model.name = args.model
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.device:
        config.model.device = args.device

    data_yaml_path = Path(args.data) if args.data else Path(cfg_dict.get("data", "data/dataset.yaml"))
    if not data_yaml_path.exists():
        logger.error(f"Dataset YAML configuration not found: {data_yaml_path}")
        sys.exit(1)

    logger.info(
        f"Starting training run: Model={config.model.name}, Epochs={config.training.epochs}, Batch={config.training.batch_size}, Device={config.model.device}"
    )

    trainer = VeyraxisTrainer(config=config, data_yaml=data_yaml_path)
    try:
        metrics = trainer.train(resume=args.resume)
        logger.info(f"Training finalized successfully! Summary metrics: {metrics}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
