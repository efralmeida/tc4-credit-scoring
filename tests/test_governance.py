"""Testes do catalogo, da retencao, do vies e da decomposicao."""

import pandas as pd
from credit_scoring.governance.bias import audit_slices
from credit_scoring.governance.catalog import data_catalog, legal_bases
from credit_scoring.governance.causal import (
    intervention_contrast,
    kitagawa_decomposition,
    mix_standardization,
)
from credit_scoring.governance.retention import retention_manifest


def test_catalog_excludes_identifier_and_does_not_call_features_sensitive() -> None:
    catalog = {row["column"]: row for row in data_catalog()}
    assert catalog["zip_code"]["decision"] == "excluido na leitura"
    assert catalog["annual_inc"]["sensitive_personal_data"] == "nao"
    assert catalog["purpose=medical"]["sensitive_personal_data"] == "nao"
    assert any("art. 7, V" in row["basis"] for row in legal_bases())


def test_retention_does_not_store_rejected_payload_or_delete_files() -> None:
    manifest = retention_manifest()
    assert manifest["deletes_files"] is False
    rejected = next(item for item in manifest["schedule"] if item["path"] == "nao persistido")
    assert rejected["repository_retention"] == "zero"
    events = next(item for item in manifest["schedule"] if "prediction_events" in item["path"])
    assert events["git"] == "ignorado"


def test_kitagawa_attributes_mix_when_rates_stay_constant() -> None:
    reference_scores = pd.Series([0.1] * 50 + [0.9] * 50)
    reference_target = pd.Series([0] * 50 + [1] * 50)
    current_scores = pd.Series([0.1] * 20 + [0.9] * 80)
    current_target = pd.Series([0] * 20 + [1] * 80)
    result = kitagawa_decomposition(
        reference_scores,
        reference_target,
        current_scores,
        current_target,
        bands=2,
    )
    assert abs(result["observed_delta"] - 0.3) < 1e-9
    assert abs(result["residual"]) < 1e-9
    assert abs(result["rate_effect"]) < 1e-9
    assert abs(result["composition_effect"] - 0.3) < 1e-9


def test_standardization_keeps_gap_when_mix_is_not_the_cause() -> None:
    reference_group = pd.Series(["Individual"] * 80 + ["Joint App"] * 20)
    current_group = pd.Series(["Individual"] * 50 + ["Joint App"] * 50)
    reference_target = pd.Series([0] * 90 + [1] * 10)
    current_target = pd.Series([0] * 50 + [1] * 50)
    result = mix_standardization(
        reference_group,
        reference_target,
        current_group,
        current_target,
    )
    assert result["gap_if_mix_unchanged"] is not None
    assert result["gap_if_mix_unchanged"] > 0


def test_intervention_moves_score_and_not_the_label() -> None:
    natural_scores = pd.Series([0.4] * 30 + [0.6] * 30)
    drifted_scores = pd.Series([0.7] * 30 + [0.9] * 30)
    labels = pd.Series([0] * 45 + [1] * 15)
    result = intervention_contrast(natural_scores, drifted_scores, labels, labels.copy())
    assert result["score_mean_delta"] > 0
    assert result["label_rate_delta"] == 0.0
    assert result["score_psi"] > 0.25


def test_bias_signal_is_monitoring_not_a_decision() -> None:
    groups = pd.DataFrame(
        {
            "purpose": ["debt_consolidation"] * 40 + ["small_business"] * 20,
            "term": ["36 months"] * 60,
        }
    )
    scores = pd.Series([0.2] * 40 + [0.9] * 20)
    result = audit_slices(groups, scores, cutoff=0.5, min_group_size=10, columns=["purpose"])
    assert result["cutoff_meaning"].startswith("Corte de monitoramento")
    assert result["signals"][0]["group"] == "small_business"
    assert result["signals"][0]["disparate_impact"] < 0.8
    assert "discriminacao" not in result["signals"][0]["meaning"] or "Nao e prova" in result["signals"][0]["meaning"]
