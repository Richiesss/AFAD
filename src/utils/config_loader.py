from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    YAML設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス

    Returns:
        Dict[str, Any]: 設定内容の辞書

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        yaml.YAMLError: パースエラーの場合
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config file: {e}") from e
