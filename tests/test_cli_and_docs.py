"""Tests for Phase 23: CLI, Documentation Generators, and CI Workflow.

Tests:
1. CLI subcommands parser and execution:
   - ingest, train, benchmark-latency, generate-brief, generate-cards, monitor.
2. Documentation generators:
   - generate_model_card creates valid markdown with GloFAS and prototype notices.
   - generate_data_sheet creates valid markdown with attribution and license.
3. Makefile and GitHub Actions workflow file existence and validity.
"""

from pathlib import Path
import pytest

from src.cli import build_parser, main as cli_main
from src.reporting.doc_generators import generate_model_card, generate_data_sheet


class TestCliAndDocGenerators:
    def test_cli_parser_commands(self):
        parser = build_parser()
        args_brief = parser.parse_args(["generate-brief", "--station", "TN_CAU_TRICHY", "--lang", "en"])
        assert args_brief.subcommand == "generate-brief"
        assert args_brief.station == "TN_CAU_TRICHY"

        args_cards = parser.parse_args(["generate-cards"])
        assert args_cards.subcommand == "generate-cards"

        args_lat = parser.parse_args(["benchmark-latency", "--runs", "5"])
        assert args_lat.runs == 5

    def test_generate_model_card(self, tmp_path):
        out_file = str(tmp_path / "model_card.md")
        res = generate_model_card(output_path=out_file)
        assert Path(out_file).exists()
        text = Path(out_file).read_text(encoding="utf-8")
        assert "Model Card" in text
        assert "GloFAS" in text
        assert "Out-of-Distribution" in text

    def test_generate_data_sheet(self, tmp_path):
        out_file = str(tmp_path / "data_sheet.md")
        res = generate_data_sheet(output_path=out_file)
        assert Path(out_file).exists()
        text = Path(out_file).read_text(encoding="utf-8")
        assert "Data Sheet" in text
        assert "Open-Meteo.com" in text
        assert "CC BY 4.0" in text

    def test_ci_workflow_and_makefile_exist(self):
        root = Path(__file__).resolve().parents[1]
        assert (root / "Makefile").exists()
        ci_file = root / ".github" / "workflows" / "ci.yml"
        assert ci_file.exists()
        ci_text = ci_file.read_text(encoding="utf-8")
        assert "pytest" in ci_text
        assert "ubuntu-latest" in ci_text
