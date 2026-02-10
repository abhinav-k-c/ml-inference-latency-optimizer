import yaml
from pathlib import Path
from config.schema import AppConfig

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config() -> AppConfig:
    with open(CONFIG_PATH, "r") as f:
        raw_config = yaml.safe_load(f)

    return AppConfig.model_validate(raw_config)
