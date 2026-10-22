"""Reproducibility utilities for HelpDeskNet-50K.

Provides functionality for deterministic execution, environment auditing,
configuration persistence, and data integrity verification via SHA256 checksums.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

logger = logging.getLogger(__name__)

DEFAULT_PACKAGES: List[str] = [
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "plotly",
    "streamlit",
    "cleanlab",
    "pyyaml",
    "joblib",
    "pytest",
    "jupyter",
]


def set_global_seeds(seed: int = 42) -> None:
    """Set global random seeds across Python, NumPy, and PyTorch (if available).

    Ensures reproducible results across all data splitting, noise simulation,
    and model training stages.

    Args:
        seed: Non-negative integer random seed.

    Raises:
        TypeError: If seed is not an integer.
        ValueError: If seed is negative.
    """
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Attempt to seed PyTorch if installed
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug("PyTorch random seed set to %d (CUDA deterministic: True)", seed)
    except ImportError:
        logger.debug("PyTorch is not installed; skipping PyTorch seed configuration")

    logger.info("Global random seed successfully set to %d", seed)


def get_package_versions(
    packages: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Retrieve installed versions of critical libraries for audit logging.

    Args:
        packages: Optional list of package names to inspect. If None,
            the default HelpDeskNet-50K stack is audited.

    Returns:
        Dictionary mapping package/runtime names to version strings.
    """
    pkg_list = packages if packages is not None else DEFAULT_PACKAGES
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }

    for pkg in pkg_list:
        version_str = "not installed"
        # Try importlib.metadata first (standard in Python 3.8+)
        try:
            version_str = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            # Fallback: normalize dashes/underscores
            alt_name = pkg.replace("-", "_") if "-" in pkg else pkg.replace("_", "-")
            try:
                version_str = importlib.metadata.version(alt_name)
            except importlib.metadata.PackageNotFoundError:
                # Fallback: attempt importing module
                try:
                    mod = importlib.import_module(pkg.replace("-", "_"))
                    version_str = getattr(mod, "__version__", "unknown")
                except ImportError:
                    version_str = "not installed"

        versions[pkg] = version_str

    logger.debug("Package versions audited: %s", versions)
    return versions


def save_experiment_config(
    config: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """Save an experiment or dataset configuration dictionary to YAML or JSON.

    Creates any missing parent directories automatically.

    Args:
        config: Configuration dictionary to serialize.
        path: Destination file path (YAML or JSON format).

    Raises:
        TypeError: If config is not a dictionary.
        ValueError: If file extension is unsupported.
        IOError: If writing to destination fails.
    """
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dictionary, got {type(config).__name__}")

    target_path = Path(path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = target_path.suffix.lower()
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            if suffix in [".yaml", ".yml"]:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif suffix == ".json":
                json.dump(config, f, indent=2)
            else:
                raise ValueError(
                    f"Unsupported configuration format '{suffix}'. Use .yaml, .yml, or .json"
                )
        logger.info("Saved configuration successfully to %s", target_path)
    except IOError as err:
        logger.error("Failed to save configuration to %s: %s", target_path, err)
        raise


def load_experiment_config(
    path: Union[str, Path],
) -> Dict[str, Any]:
    """Load an experiment or dataset configuration file from YAML or JSON.

    Args:
        path: Path to configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If file format is invalid or file does not contain a dictionary.
        IOError: If reading the file fails.
    """
    target_path = Path(path).resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {target_path}")

    suffix = target_path.suffix.lower()
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            if suffix in [".yaml", ".yml"]:
                loaded = yaml.safe_load(f)
            elif suffix == ".json":
                loaded = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration format '{suffix}'. Use .yaml, .yml, or .json"
                )

        if loaded is None:
            loaded = {}
        elif not isinstance(loaded, dict):
            raise ValueError(
                f"Configuration at {target_path} must be a dictionary, got {type(loaded).__name__}"
            )

        logger.info("Loaded configuration successfully from %s", target_path)
        return loaded
    except (yaml.YAMLError, json.JSONDecodeError) as err:
        logger.error("Failed to parse configuration at %s: %s", target_path, err)
        raise ValueError(f"Malformed configuration file {target_path}: {err}") from err
    except IOError as err:
        logger.error("Failed to read configuration at %s: %s", target_path, err)
        raise


def compute_file_checksum(
    filepath: Union[str, Path],
    algorithm: str = "sha256",
    chunk_size: int = 65536,
) -> str:
    """Compute the cryptographic hash (SHA256 by default) of a file.

    Streams content in chunks to ensure memory efficiency on large datasets.

    Args:
        filepath: Path to the target file.
        algorithm: Hash algorithm name supported by hashlib (default: 'sha256').
        chunk_size: Number of bytes to read per chunk (default: 65,536 / 64 KB).

    Returns:
        Hexadecimal hash string.

    Raises:
        FileNotFoundError: If filepath does not exist.
        IsADirectoryError: If filepath points to a directory.
        ValueError: If algorithm is invalid or chunk_size <= 0.
        IOError: If an error occurs during file reading.
    """
    target_path = Path(filepath).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"File not found for checksum calculation: {target_path}")
    if target_path.is_dir():
        raise IsADirectoryError(f"Cannot compute checksum on directory: {target_path}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as err:
        raise ValueError(f"Unsupported hashing algorithm '{algorithm}': {err}") from err

    try:
        with open(target_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        logger.debug("Computed %s for %s: %s", algorithm, target_path.name, digest)
        return digest
    except IOError as err:
        logger.error("Error reading file %s during checksum: %s", target_path, err)
        raise
