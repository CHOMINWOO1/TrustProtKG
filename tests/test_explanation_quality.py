import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from trustprotkg.explanation_quality import (
    build_explanation_card,
    render_explanation_card_markdown,
    run_explanation_quality_evaluation,
    score_explanation_quality,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "configs" / "v0.7_explanation_quality.yaml"


def _example_explanation() -> dict:
    return {
        "variant": {"id": "Variant:V_TEST", "variant_id": "V_TEST"},
        "residue": {
            "id": "Residue:TEST:A:10",
            "protein_id": "TEST_HUMAN",
            "chain_id": "A",
            "residue_index": 10,
            "plddt": 90.0,
        },
        "features": {
            "reference_aa": "R",
            "alternate_aa": "H",
            "clinical_label": "pathogenic",
        },
        "evidence": {
            "located_at": [
                {
                    "edge_type": "LOCATED_AT",
                    "provenance": {
                        "source_db": "ClinVar fixture",
                        "evidence_type": "clinical_assertion",
                        "confidence": 0.8,
                    },
                }
            ],
            "structural_contacts": [
                {
                    "edge_type": "STRUCTURAL_CONTACT",
                    "distance": 4.0,
                    "confidence": 0.9,
                    "contact_residue": {
                        "id": "Residue:TEST:A:11",
                        "residue_name": "GLY",
                    },
                    "provenance": {
                        "source_db": "AlphaFold fixture",
                        "evidence_type": "CA_distance_le_8_angstrom",
                        "confidence": 0.9,
                    },
                }
            ],
            "domain_membership": [
                {
                    "edge_type": "LOCATED_AT",
                    "domain": {"domain_id": "IPR_TEST", "name": "test domain"},
                    "provenance": {
                        "source_db": "InterPro fixture",
                        "evidence_type": "domain",
                        "confidence": 0.7,
                    },
                }
            ],
            "protein_context": [
                {
                    "edge_type": "HAS_FUNCTION",
                    "neighbor": {"id": "GO_Term:GO_TEST", "name": "test function"},
                    "provenance": {
                        "source_db": "GO fixture",
                        "evidence_type": "IDA",
                        "confidence": 0.8,
                    },
                },
                {
                    "edge_type": "PARTICIPATES_IN",
                    "neighbor": {"id": "Pathway:TEST", "name": "test pathway"},
                    "provenance": {
                        "source_db": "Reactome fixture",
                        "evidence_type": "TAS",
                        "confidence": 0.8,
                    },
                },
                {
                    "edge_type": "ASSOCIATED_WITH",
                    "neighbor": {"id": "Disease:TEST", "name": "test disease"},
                    "provenance": {
                        "source_db": "Disease fixture",
                        "evidence_type": "literature",
                        "confidence": 0.8,
                    },
                },
            ],
        },
    }


def test_explanation_quality_metrics_cover_all_required_components():
    score = score_explanation_quality(_example_explanation(), source_diversity_cap=5)

    assert score["variant_id"] == "V_TEST"
    assert score["provenance_path_coverage"] == 1.0
    assert score["evidence_source_diversity_score"] == 1.0
    assert score["structural_contact_plausibility_score"] == 0.95
    assert score["domain_support_score"] == 0.7
    assert score["go_pathway_disease_support_score"] == 1.0
    assert 0.0 < score["explanation_quality_score"] <= 1.0


def test_explanation_card_contains_reviewer_facing_contexts():
    explanation = _example_explanation()
    score = score_explanation_quality(explanation, source_diversity_cap=5)
    card = build_explanation_card(
        explanation,
        score,
        [
            {
                "run_name": "random_split",
                "split_strategy": "existing_split",
                "holdout_protein_id": "",
                "feature_set": "structure_plus_kg",
                "clinical_label": "pathogenic",
                "label_binary": 1,
                "prob_pathogenic": 0.91,
                "predicted_label": 1,
                "prediction_correct": True,
            }
        ],
    )
    markdown = render_explanation_card_markdown(card)

    assert card["variant_id"] == "V_TEST"
    assert card["amino_acid_change"] == "R10H"
    assert card["evidence_summary"]["domains"][0]["domain_id"] == "IPR_TEST"
    assert "Prediction Contexts" in markdown
    assert "structure_plus_kg" in markdown


def test_v07_explanation_quality_run_writes_report_scores_and_cards():
    result = run_explanation_quality_evaluation(CONFIG)

    assert result.variant_count == 9
    assert result.card_count == 3
    for path in [
        result.variant_quality_scores,
        result.quality_by_context,
        result.explanation_cards_jsonl,
        result.explanation_cards_markdown,
        result.metadata,
        result.report,
    ]:
        assert path.exists()

    variant_quality = pd.read_csv(result.variant_quality_scores)
    assert {
        "provenance_path_coverage",
        "evidence_source_diversity_score",
        "structural_contact_plausibility_score",
        "domain_support_score",
        "go_pathway_disease_support_score",
        "explanation_quality_score",
    }.issubset(variant_quality.columns)
    assert set(variant_quality["protein_id"]) == {
        "TP53_HUMAN",
        "BRCA1_HUMAN",
        "PTEN_HUMAN",
    }

    quality_by_context = pd.read_csv(result.quality_by_context)
    assert set(quality_by_context["split_strategy"]) == {
        "existing_split",
        "leave_one_protein_out",
    }
    assert set(quality_by_context["feature_set"]) == {
        "structure_only",
        "kg_only",
        "structure_plus_kg",
        "no_domain",
        "no_structural_contacts",
    }
    assert "mean_feature_evidence_alignment_score" in quality_by_context.columns

    cards = [
        json.loads(line)
        for line in result.explanation_cards_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert {card["protein_id"] for card in cards} == {
        "TP53_HUMAN",
        "BRCA1_HUMAN",
        "PTEN_HUMAN",
    }
    assert all(card["prediction_contexts"] for card in cards)

    report = result.report.read_text(encoding="utf-8")
    for section in [
        "## Explanation-Quality Metric Definitions",
        "## Per-Variant Explanation Quality",
        "## Quality By Split And Feature Context",
        "## Case Study Cards",
        "## Limitations",
    ]:
        assert section in report

    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["trustprotkg_version"] == "41.0.0"
    assert metadata["dataset_summary"]["case_study_cards"] == 3


def test_v07_explanation_quality_cli_executes():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trustprotkg.explanation_quality",
            "--config",
            str(CONFIG),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "TrustProtKG explanation-quality evaluation complete" in completed.stdout







































