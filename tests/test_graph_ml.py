import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from trustprotkg.graph_ml import run_graph_ml_experiment


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "configs" / "v2.0_graph_ml.yaml"
RUN_DIR = ROOT / "experiments" / "runs" / "v2.0_graph_ml"


def test_v20_graph_ml_baseline_generates_spectral_embedding_comparison():
    result = run_graph_ml_experiment(CONFIG)

    assert result.run_dir == RUN_DIR
    assert result.variant_count == 24
    assert result.split_run_count == 13
    assert result.feature_set_count == 3
    for path in [
        result.feature_table,
        result.metrics,
        result.predictions,
        result.comparison,
        result.embedding_summary,
        result.metadata,
        result.report,
    ]:
        assert path.exists()

    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["trustprotkg_version"] == "41.0.0"
    assert metadata["dataset_summary"]["embedding_dimensions"] == 8
    assert metadata["dataset_summary"]["metric_rows"] == 39

    features = pd.read_csv(result.feature_table)
    embedding_columns = [column for column in features.columns if column.startswith("graph_ml_emb_")]
    assert len(embedding_columns) == 8
    assert features[embedding_columns].abs().sum().sum() > 0

    metrics = pd.read_csv(result.metrics)
    assert {
        "transparent_structure_plus_kg",
        "spectral_graph_embedding",
        "transparent_plus_spectral",
    } == set(metrics["feature_set"])

    comparison = pd.read_csv(result.comparison)
    assert "delta_f1_vs_transparent" in comparison.columns
    assert comparison.shape[0] == 39


def test_v20_graph_ml_cli_writes_report():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trustprotkg.graph_ml",
            "--config",
            str(CONFIG),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "TrustProtKG graph ML baseline complete" in completed.stdout
    report = (RUN_DIR / "graph_ml_report.md").read_text(encoding="utf-8")
    assert "## Mean Performance By Feature Set" in report
    assert "Interpretation Guardrail" in report







































