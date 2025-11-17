"""
Utility functions for crypto derivatives research.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages research configuration.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded config from {path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def save_config(self, path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved config to {path}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value


class DataPersistence:
    """
    Helper for saving and loading data in various formats.
    """

    @staticmethod
    def save_dataframe(df: pd.DataFrame, path: str, format: str = "csv") -> None:
        """
        Save DataFrame to disk.

        Args:
            df: DataFrame to save
            path: Output path
            format: 'csv', 'parquet', or 'feather'
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if format == "csv":
            df.to_csv(path, index=False)
        elif format == "parquet":
            df.to_parquet(path, index=False)
        elif format == "feather":
            df.to_feather(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {len(df)} rows to {path}")

    @staticmethod
    def load_dataframe(path: str, format: str = "csv") -> pd.DataFrame:
        """
        Load DataFrame from disk.

        Args:
            path: Input path
            format: 'csv', 'parquet', or 'feather'

        Returns:
            Loaded DataFrame
        """
        if format == "csv":
            df = pd.read_csv(path)
        elif format == "parquet":
            df = pd.read_parquet(path)
        elif format == "feather":
            df = pd.read_feather(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Loaded {len(df)} rows from {path}")
        return df


class ExperimentLogger:
    """
    Logs experimental results for tracking.
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log_experiment(self, experiment_name: str, results: Dict[str, Any]) -> str:
        """
        Log experiment results to JSON file.

        Args:
            experiment_name: Name of experiment
            results: Dictionary of results

        Returns:
            Path to log file
        """
        log_path = os.path.join(self.log_dir, f"{experiment_name}.json")

        with open(log_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Logged experiment to {log_path}")
        return log_path


def ensure_directories(base_path: str = "./data") -> Dict[str, str]:
    """
    Ensure all required data directories exist.

    Args:
        base_path: Base data directory

    Returns:
        Dictionary mapping directory names to paths
    """
    dirs = {
        "raw": os.path.join(base_path, "raw"),
        "processed": os.path.join(base_path, "processed"),
        "ohlcv": os.path.join(base_path, "raw", "ohlcv"),
        "funding": os.path.join(base_path, "raw", "funding"),
        "liquidations": os.path.join(base_path, "raw", "liquidations"),
        "oi": os.path.join(base_path, "raw", "oi"),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    logger.info(f"Ensured {len(dirs)} directories exist in {base_path}")
    return dirs
