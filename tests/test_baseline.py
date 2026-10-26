"""Testes do baseline e do bloqueio de inferencia."""

import numpy as np
import pandas as pd
import pytest
from credit_scoring.data.schema import CONTRACT_ERRORS, example_scoring_row, scoring_frame
from credit_scoring.models.baseline import (
    predict_default_probability,
    train_baseline,
)


def _training_frame(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    payload = []
    for index in range(rows):
        defaulted = index % 3 == 0
        fico = 640.0 if defaulted else 740.0
        dti = 45.0 if defaulted else 12.0
        status = "Charged Off" if defaulted else "Fully Paid"
        payload.append(
            example_scoring_row(
                loan_amnt=float(5_000 + (index % 10) * 500),
                annual_inc=40_000.0 if defaulted else 90_000.0,
                dti=float(np.clip(dti + rng.normal(0, 0.5), 0, 60)),
                fico_range_low=fico,
                loan_status=status,
                issue_d=pd.Timestamp("2014-01-01") + pd.DateOffset(months=index % 12),
            )
        )
    return pd.DataFrame(payload)


def test_training_requires_valid_reference() -> None:
    frame = _training_frame(40)
    frame.loc[0, "dti"] = 999.0
    with pytest.raises(CONTRACT_ERRORS):
        train_baseline(frame)


def test_baseline_learns_signal_and_blocks_bad_inference() -> None:
    artifact = train_baseline(_training_frame())
    assert artifact["metrics"]["roc_auc"] > 0.8
    assert artifact["metrics"]["ks"] > 0.5

    scores = predict_default_probability(artifact, scoring_frame())
    assert scores.between(0, 1).all()

    invalid = scoring_frame([example_scoring_row(term="12 months")])
    with pytest.raises(CONTRACT_ERRORS):
        predict_default_probability(artifact, invalid)
