"""
tests/test_model_artifacts.py
─────────────────────────────
Part A Step 6 test:
Ensures that every model option offered in the dashboard dropdown
has a corresponding trained artifact on disk for all trained stations and horizons.
Fails if any offered model has a missing artifact.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def get_dashboard_models_list() -> list[str]:
    """Extract models_list from src/dashboard/app.py using AST to avoid Streamlit bare-mode side effects."""
    app_path = PROJECT_ROOT / "src" / "dashboard" / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "models_list":
                    return [elt.value for elt in node.value.elts]
    raise ValueError("Could not find 'models_list' definition in src/dashboard/app.py")


class TestDashboardModelArtifacts:
    def test_dashboard_models_have_trained_artifacts(self):
        """Fail if the dashboard offers a model option for which no artifact exists."""
        from src.utils.config import load_data_config

        models_list = get_dashboard_models_list()
        assert len(models_list) > 0, "No models defined in dashboard models_list!"

        cfg = load_data_config()
        models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]

        # Only check stations that have trained models (excluding unmodeled/insufficient stations)
        all_stations = cfg.get("stations", [])
        trained_stations = []
        for s in all_stations:
            sid = s["id"]
            s_dir = models_dir / sid
            if s_dir.exists() and len(list(s_dir.glob("*.joblib"))) > 0:
                trained_stations.append(sid)

        assert len(trained_stations) > 0, "No trained stations found in models directory!"

        missing_artifacts = []
        for sid in trained_stations:
            s_dir = models_dir / sid
            for horizon in [1, 3, 7]:
                for mod_name in models_list:
                    # Check candidate filenames supported by dashboard loader
                    candidate_paths = [
                        s_dir / f"{sid}_{mod_name}_h{horizon}d.joblib",
                        s_dir / f"{mod_name}_h{horizon}d.joblib",
                        models_dir / f"{sid}_{mod_name}_h{horizon}d.joblib",
                    ]
                    if not any(p.exists() for p in candidate_paths):
                        missing_artifacts.append(f"Station {sid}, Horizon {horizon}d, Model '{mod_name}'")

        if missing_artifacts:
            pytest.fail(
                f"Dashboard offers model options without corresponding saved artifacts:\n"
                + "\n".join(missing_artifacts[:15])
                + (f"\n... and {len(missing_artifacts) - 15} more" if len(missing_artifacts) > 15 else "")
            )
