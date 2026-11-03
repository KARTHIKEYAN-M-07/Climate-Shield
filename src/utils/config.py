# =============================================================================
# src/utils/config.py  — shared config loader
# =============================================================================
"""Utility: load YAML config files and expose typed helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a plain dict.

    Parameters
    ----------
    path:
        Absolute or relative (to project root) path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML contents.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    yaml.YAMLError
        If the file cannot be parsed.
    """
    resolved = Path(path) if Path(path).is_absolute() else _PROJECT_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    logger.debug("Loaded config from %s", resolved)
    return data or {}


def load_data_config() -> dict[str, Any]:
    """Load configs/data.yaml."""
    return load_yaml("configs/data.yaml")


def load_stations_config() -> dict[str, Any]:
    """Load configs/stations.yaml."""
    return load_yaml("configs/stations.yaml")


def load_all_stations() -> list[dict[str, Any]]:
    """Return merged list of all stations from data.yaml and stations.yaml."""
    cfg = load_data_config()
    stns = list(cfg.get("stations", []))
    existing_ids = {s["id"] for s in stns}

    stn_path = _PROJECT_ROOT / "configs" / "stations.yaml"
    if stn_path.exists():
        stn_cfg = load_stations_config()
        for s in stn_cfg.get("stations", []):
            if s["id"] not in existing_ids:
                stns.append(s)
                existing_ids.add(s["id"])
    return stns


def load_thresholds_config() -> dict[str, Any]:
    """Load configs/thresholds.yaml."""
    return load_yaml("configs/thresholds.yaml")


def get_station(config: dict[str, Any], station_id: str | None = None) -> dict[str, Any]:
    """Return the station dict matching *station_id*.

    Falls back to the ``default_station_id`` if *station_id* is None.

    Raises
    ------
    KeyError
        If the station id is not found in the config.
    """
    sid = station_id or config["default_station_id"]
    for station in config.get("stations", []):
        if station["id"] == sid:
            return station
    available = [s["id"] for s in config.get("stations", [])]
    raise KeyError(
        f"Station '{sid}' not found in config. Available: {available}"
    )
