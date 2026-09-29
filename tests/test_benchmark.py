import json
from pathlib import Path

import pandas as pd

from trustprotkg.evaluation import FEATURE_SETS, run_baseline_evaluation
from trustprotkg.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmark.yaml"


def test_benchmark_pipeline_writes_evaluation_and_explanations():
    result = run_pipeline(CONFIG)

    assert result.node_count > 30
    assert result.edge_count > 100
    assert result.baseline_metrics is not None
    assert result.baseline_predictions is not None
    assert result.explanations is not None
    assert result.baseline_metrics.exists()
    assert result.baseline_predictions.exists()
    assert result.explanations.exists()

    features = pd.read_csv(result.variant_features)
    metrics = pd.read_csv(result.baseline_metrics)
    predictions = pd.read_csv(result.baseline_predictions)
    explanations = [
        json.loads(line)
        for line in result.explanations.read_text(encoding="utf-8").splitlines()
    ]

    assert set(features["protein_id"]) == {"TP53_HUMAN", "BRCA1_HUMAN", "PTEN_HUMAN"}
    assert features.shape[0] == 9
    assert set(metrics["feature_set"]) == set(FEATURE_SETS)
    assert set(predictions["feature_set"]) == set(FEATURE_SETS)
    assert len(explanations) == features.shape[0]
    first = explanations[0]
    assert first["evidence"]["located_at"][0]["provenance"]["source_db"]
    assert first["evidence"]["structural_contacts"]
    assert first["evidence"]["protein_context"]


def test_baseline_evaluation_uses_train_test_split():
    feature_table = pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3", "v4"],
            "protein_id": ["p1", "p1", "p2", "p2"],
            "clinical_label": ["benign", "pathogenic", "benign", "pathogenic"],
            "label_binary": [0, 1, 0, 1],
            "split": ["train", "train", "test", "test"],
            "residue_confidence": [90.0, 70.0, 88.0, 72.0],
            "num_structural_contacts": [2, 6, 3, 7],
            "avg_contact_distance": [6.0, 4.0, 6.2, 4.1],
            "is_in_domain": [False, True, False, True],
            "num_go_pathway_disease_edges": [2, 4, 2, 4],
        }
    )

    result = run_baseline_evaluation(feature_table)

    assert set(result.metrics["feature_set"]) == set(FEATURE_SETS)
    assert set(result.predictions["variant_id"]) == {"v3", "v4"}
    assert result.predictions["prob_pathogenic"].between(0.0, 1.0).all()
















