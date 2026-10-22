"""Unit tests for reproducibility and configuration utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.utils.reproducibility import (
    compute_file_checksum,
    get_package_versions,
    load_experiment_config,
    save_experiment_config,
    set_global_seeds,
)


def test_set_global_seeds_valid() -> None:
    """Test seed setting runs without error for valid integers."""
    set_global_seeds(42)
    set_global_seeds(0)


def test_set_global_seeds_invalid_type() -> None:
    """Test TypeError is raised when seed is not an integer."""
    with pytest.raises(TypeError, match="Seed must be an integer"):
        set_global_seeds("42")  # type: ignore[arg-type]


def test_set_global_seeds_negative() -> None:
    """Test ValueError is raised when seed is negative."""
    with pytest.raises(ValueError, match="Seed must be non-negative"):
        set_global_seeds(-1)


def test_get_package_versions_default() -> None:
    """Test package version auditing returns expected keys."""
    versions = get_package_versions()
    assert "python" in versions
    assert "platform" in versions
    assert "pandas" in versions
    assert "scikit-learn" in versions
    assert versions["pandas"] != "not installed"


def test_get_package_versions_custom() -> None:
    """Test package version auditing with custom list."""
    versions = get_package_versions(["pandas", "non_existent_package_xyz"])
    assert "pandas" in versions
    assert versions["non_existent_package_xyz"] == "not installed"


def test_save_and_load_config_yaml(tmp_path: Path) -> None:
    """Test saving and loading YAML configuration."""
    cfg = {"experiment": {"name": "test_exp", "random_seed": 42}}
    file_path = tmp_path / "subdir" / "config.yaml"

    save_experiment_config(cfg, file_path)
    assert file_path.exists()

    loaded = load_experiment_config(file_path)
    assert loaded == cfg


def test_save_and_load_config_json(tmp_path: Path) -> None:
    """Test saving and loading JSON configuration."""
    cfg = {"dataset": {"name": "HelpDeskNet", "max_records": 1000}}
    file_path = tmp_path / "config.json"

    save_experiment_config(cfg, file_path)
    assert file_path.exists()

    loaded = load_experiment_config(file_path)
    assert loaded == cfg


def test_load_config_not_found(tmp_path: Path) -> None:
    """Test FileNotFoundError when config file is missing."""
    with pytest.raises(FileNotFoundError):
        load_experiment_config(tmp_path / "missing.yaml")


def test_save_config_invalid_type(tmp_path: Path) -> None:
    """Test TypeError when saving non-dictionary."""
    with pytest.raises(TypeError, match="Config must be a dictionary"):
        save_experiment_config(["not", "a", "dict"], tmp_path / "test.yaml")  # type: ignore[arg-type]


def test_compute_file_checksum(tmp_path: Path) -> None:
    """Test computing checksum for known text."""
    test_file = tmp_path / "sample.txt"
    test_file.write_bytes(b"hello world\n")

    # echo "hello world\n" sha256
    # hashlib.sha256(b"hello world\n").hexdigest()
    import hashlib

    expected = hashlib.sha256(b"hello world\n").hexdigest()
    result = compute_file_checksum(test_file)
    assert result == expected


def test_compute_file_checksum_not_found(tmp_path: Path) -> None:
    """Test FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError):
        compute_file_checksum(tmp_path / "non_existent.txt")


def test_compute_file_checksum_directory(tmp_path: Path) -> None:
    """Test IsADirectoryError when passing a directory."""
    with pytest.raises(IsADirectoryError):
        compute_file_checksum(tmp_path)
