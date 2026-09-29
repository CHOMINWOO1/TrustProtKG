import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from trustprotkg.external_validation import run_external_validation
from trustprotkg.graph_ml import run_graph_ml_experiment
from trustprotkg.model_calibration import run_model_calibration


ROOT = Path(__file__).resolve().parents[1]
GRAPH_CONFIG = ROOT / "experiments" / "configs" / "v2.0_graph_ml.yaml"
VALIDATION_CONFIG = ROOT / "experiments" / "configs" / "v2.1_external_validation.yaml"
CONFIG = ROOT / "experiments" / "configs" / "v2.5_model_calibration.yaml"
RUN_DIR = ROOT / "paper" / "v2.5"


def test_v25_model_calibration_generates_bins_thresholds_and_balance_tables():
    run_graph_ml_experiment(GRAPH_CONFIG)
    run_external_validation(VALIDATION_CONFIG)
    result = run_model_calibration(CONFIG)

    assert result.run_dir == RUN_DIR
    assert result.prediction_count == 243
    assert result.threshold_row_count == 54
    assert result.decision_row_count == 6
    for path in [
        result.combined_predictions,
        result.calibration_bins,
        result.calibration_summary,
        result.threshold_sweep,
        result.decision_thresholds,
        result.class_balance,
        result.metadata,
        result.report,
        result.readme,
    ]:
        assert path.exists()

    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["trustprotkg_version"] == "41.0.0"
    assert metadata["counts"]["contexts"] == 2
    assert metadata["counts"]["feature_sets"] == 3
    assert metadata["counts"]["calibration_bin_rows"] == 30

    calibration = pd.read_csv(result.calibration_summary)
    assert set(calibration["evaluation_context"]) == {
        "internal_family_aware",
        "external_validation_snapshot",
    }
    assert calibration["expected_calibration_error"].between(0.0, 1.0).all()

    thresholds = pd.read_csv(result.decision_thresholds)
    assert thresholds.shape[0] == 6
    assert thresholds["best_f1_threshold"].between(0.1, 0.9).all()
    assert thresholds["minority_fraction"].gt(0).all()

    report = result.report.read_text(encoding="utf-8")
    assert "## Calibration Summary" in report
    assert "## Decision Threshold Summary" in report
    assert "Interpretation Guardrail" in report


def test_v25_model_calibration_cli_writes_report():
    run_graph_ml_experiment(GRAPH_CONFIG)
    run_external_validation(VALIDATION_CONFIG)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trustprotkg.model_calibration",
            "--config",
            str(CONFIG),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "TrustProtKG model calibration complete" in completed.stdout
    assert (RUN_DIR / "model_calibration_report.md").exists()







































