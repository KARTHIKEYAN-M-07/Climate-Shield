"""Model Registry with versioning, data/config hashes, and champion/challenger promotion.

Rules:
- Records model version, creation date, data hash, config hash, git commit.
- Promotion rule: challenger is only promoted to champion if it beats champion on validation
  AND passes the statistical significance test (e.g. Diebold-Mariano p < 0.05).
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """Manages models/registry.json for tracking and promoting models."""

    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_registry_exists()

    def _ensure_registry_exists(self) -> None:
        if not self.registry_path.exists():
            initial_data = {
                "schema_version": "1.0",
                "champion": None,
                "models": {},
            }
            self._write_registry(initial_data)

    def _read_registry(self) -> Dict[str, Any]:
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_registry(self, data: Dict[str, Any]) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def compute_file_hash(path: str) -> str:
        """Compute SHA-256 hash of a file."""
        p = Path(path)
        if not p.exists():
            return "file_not_found"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def get_git_commit() -> str:
        """Retrieve current git commit hash, or 'unversioned'."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except Exception:
            return "unversioned"

    def register_model(
        self,
        model_name: str,
        version: str,
        station_id: str,
        horizon_days: int,
        validation_metrics: Dict[str, float],
        data_hash: str,
        config_hash: str,
        is_champion: bool = False,
    ) -> Dict[str, Any]:
        """Register a new model entry."""
        data = self._read_registry()
        entry_key = f"{model_name}_{version}_{station_id}_h{horizon_days}"
        entry = {
            "model_name": model_name,
            "version": version,
            "station_id": station_id,
            "horizon_days": horizon_days,
            "validation_metrics": validation_metrics,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "git_commit": self.get_git_commit(),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "status": "champion" if is_champion else "candidate",
        }
        data["models"][entry_key] = entry
        if is_champion:
            data["champion"] = entry_key

        self._write_registry(data)
        return entry

    def evaluate_and_promote(
        self,
        challenger_key: str,
        metric: str = "rmse",
        smaller_is_better: bool = True,
        dm_p_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate challenger against current champion.
        
        Promotes only if:
        1. challenger validation metric is strictly better than champion, AND
        2. dm_p_value is provided and < 0.05 (statistically significant).
        """
        data = self._read_registry()
        champion_key = data.get("champion")

        if challenger_key not in data["models"]:
            raise ValueError(f"Challenger {challenger_key} not in registry.")

        challenger = data["models"][challenger_key]

        # If no champion exists, auto-promote first valid model
        if champion_key is None or champion_key not in data["models"]:
            data["champion"] = challenger_key
            challenger["status"] = "champion"
            challenger["promotion_reason"] = "Initial champion (no previous champion)"
            self._write_registry(data)
            return {"promoted": True, "reason": "Initial champion"}

        champion = data["models"][champion_key]

        champ_score = champion.get("validation_metrics", {}).get(metric)
        chall_score = challenger.get("validation_metrics", {}).get(metric)

        if champ_score is None or chall_score is None:
            return {
                "promoted": False,
                "reason": f"Missing {metric} validation metric on champion or challenger",
            }

        metric_improved = (
            chall_score < champ_score if smaller_is_better else chall_score > champ_score
        )

        if not metric_improved:
            return {
                "promoted": False,
                "reason": f"Challenger metric ({chall_score}) does not beat champion ({champ_score})",
            }

        # Check significance test requirement
        if dm_p_value is None or dm_p_value >= 0.05:
            return {
                "promoted": False,
                "reason": (
                    f"Challenger beat champion on {metric} ({chall_score} vs {champ_score}), "
                    f"but failed significance test (p-value: {dm_p_value} >= 0.05 or missing)"
                ),
            }

        # Both passed -> promote
        champion["status"] = "retired"
        challenger["status"] = "champion"
        challenger["promotion_reason"] = (
            f"Beat champion on {metric} ({chall_score:.4f} vs {champ_score:.4f}) with p={dm_p_value:.4f}"
        )
        data["champion"] = challenger_key
        self._write_registry(data)

        return {
            "promoted": True,
            "reason": challenger["promotion_reason"],
            "new_champion": challenger_key,
        }

    def get_champion(self) -> Optional[Dict[str, Any]]:
        """Return the current champion model record."""
        data = self._read_registry()
        champ_key = data.get("champion")
        if champ_key and champ_key in data["models"]:
            return data["models"][champ_key]
        return None
