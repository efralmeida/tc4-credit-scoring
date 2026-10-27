"""Testes da simulacao de producao e da evidencia de drift."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from credit_scoring.config import INTENTIONAL_DRIFT_COLUMNS, PSI_MODERATE_THRESHOLD
from credit_scoring.data.schema import (
    CONTRACT_ERRORS,
    PRODUCTION_SCHEMA,
    example_scoring_row,
    reference_frame,
)
from credit_scoring.drift.reports import monitoring_frame, render_drift_report
from credit_scoring.drift.simulation import (
    apply_intentional_drift,
    assign_production_dates,
    score_applications,
)
from credit_scoring.drift.statistics import compare_columns, population_stability_index
from credit_scoring.models.baseline import train_baseline


def _population(rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    payload = []
    for index in range(rows):
        defaulted = index % 4 == 0
        # O sinal precisa existir no treino. Sem isso, piorar FICO e DTI
        # nao aumenta o score, e o teste deixaria de medir prediction drift.
        payload.append(
            example_scoring_row(
                loan_amnt=float(8_000 + (index % 12) * 500),
                annual_inc=float((42_000 if defaulted else 88_000) + rng.normal(0, 500)),
                dti=float(np.clip((36 if defaulted else 12) + rng.normal(0, 1.5), 1, 50)),
                fico_range_low=float(np.clip((640 if defaulted else 740) + rng.normal(0, 6), 560, 800)),
                revol_util=float(np.clip(30 + rng.normal(0, 3), 1, 80)),
                loan_status="Charged Off" if defaulted else "Fully Paid",
                issue_d=pd.Timestamp("2014-01-01") + pd.DateOffset(months=index % 12),
            )
        )
    return reference_frame(payload)


def test_production_schema_rejects_training_window() -> None:
    frame = assign_production_dates(_population(20))
    validated = PRODUCTION_SCHEMA.validate(frame, lazy=True)
    assert validated["issue_d"].min() >= pd.Timestamp("2016-01-01")

    leaked = frame.copy()
    leaked.loc[0, "issue_d"] = pd.Timestamp("2014-06-01")
    with pytest.raises(CONTRACT_ERRORS):
        PRODUCTION_SCHEMA.validate(leaked, lazy=True)


def test_intentional_drift_stays_inside_contract_and_flags_only_two_columns() -> None:
    natural = assign_production_dates(_population())
    drifted, report = apply_intentional_drift(natural)
    PRODUCTION_SCHEMA.validate(drifted, lazy=True)

    assert report["intentional_columns"] == list(INTENTIONAL_DRIFT_COLUMNS)
    assert drifted["fico_range_low"].mean() < natural["fico_range_low"].mean()
    assert drifted["dti"].mean() > natural["dti"].mean()

    comparison = compare_columns(
        natural,
        drifted,
        ["fico_range_low", "dti", "annual_inc", "revol_util"],
        ["term", "purpose"],
        intentional_columns=INTENTIONAL_DRIFT_COLUMNS,
    )
    affected = {row["column"] for row in comparison if row["affected"]}
    assert affected == set(INTENTIONAL_DRIFT_COLUMNS)
    fico = next(row for row in comparison if row["column"] == "fico_range_low")
    assert fico["psi"] >= PSI_MODERATE_THRESHOLD
    assert fico["ks_drift_detected"] is True
    unchanged = next(row for row in comparison if row["column"] == "annual_inc")
    assert unchanged["psi"] < PSI_MODERATE_THRESHOLD


def test_identical_samples_have_no_psi() -> None:
    frame = _population(80)
    psi = population_stability_index(frame["annual_inc"], frame["annual_inc"])
    assert psi < 0.01


def test_scores_move_when_fico_falls_and_dti_rises() -> None:
    reference = _population()
    artifact = train_baseline(reference)
    natural = assign_production_dates(reference)
    drifted, _ = apply_intentional_drift(natural)
    natural_scores = score_applications(artifact, natural)
    drifted_scores = score_applications(artifact, drifted)
    assert drifted_scores.mean() > natural_scores.mean()


def test_evidently_html_names_the_injected_columns(tmp_path: Path) -> None:
    natural = assign_production_dates(_population(160))
    drifted, _ = apply_intentional_drift(natural)
    scores = pd.Series(np.linspace(0.1, 0.4, len(natural)), index=natural.index)
    destination = tmp_path / "drift_report.html"
    pointer = render_drift_report(
        monitoring_frame(natural, scores),
        monitoring_frame(drifted, scores + 0.15),
        destination,
    )
    html = destination.read_text(encoding="utf-8")
    assert pointer["bytes"] > 1_000
    assert "fico_range_low" in html
    assert "dti" in html
