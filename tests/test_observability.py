"""Testes do log operacional e dos indicadores de saude."""

import json

import numpy as np
import pandas as pd
from credit_scoring.config import EXCLUDED_IDENTIFIERS, FEATURE_COLUMNS
from credit_scoring.data.schema import example_scoring_row, scoring_frame
from credit_scoring.observability.dashboard import render_observability_html
from credit_scoring.observability.logging import assert_log_minimized, score_bucket, write_jsonl
from credit_scoring.observability.metrics import summarize_batches
from credit_scoring.observability.service import score_batch


class _ConstantModel:
    """Substitui o baseline para nao treinar nos testes."""

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), 0.4)
        return np.column_stack([1.0 - positive, positive])


def _artifact() -> dict[str, object]:
    return {"pipeline": _ConstantModel(), "metadata": {"trained_at": "2026-01-01T00:00:00+00:00"}}


def _reference(rows: int = 240) -> tuple[pd.DataFrame, pd.Series]:
    payload = [
        example_scoring_row(
            fico_range_low=float(660 + (index % 20) * 5),
            dti=float(8 + (index % 15)),
            loan_amnt=float(5_000 + (index % 10) * 500),
        )
        for index in range(rows)
    ]
    return scoring_frame(payload), pd.Series(np.full(rows, 0.4), name="default_probability")


def test_score_bucket_hides_exact_probability() -> None:
    assert score_bucket(0.41) == "0.4-0.6"
    assert score_bucket(1.0) == "0.8-1.0"


def test_rejected_row_is_counted_and_not_logged(tmp_path) -> None:
    valid = example_scoring_row()
    invalid = example_scoring_row(dti=999.0)
    reference, reference_scores = _reference()
    batch, events = score_batch(
        _artifact(),
        scoring_frame([valid, invalid]),
        "probe",
        "contract_probe",
        reference,
        reference_scores,
        scored_at="2026-03-01T00:00:00+00:00",
    )
    assert batch["accepted_rows"] == 1
    assert batch["rejected_rows"] == 1
    assert batch["error_rate"] == 0.5
    assert batch["rejected_payload_stored"] is False
    assert len(events) == 1
    assert "dti" not in events[0]
    assert_log_minimized(batch)
    assert_log_minimized(events[0])
    destination = tmp_path / "events.jsonl"
    write_jsonl(destination, events)
    raw = destination.read_text(encoding="utf-8")
    for column in (*FEATURE_COLUMNS, *EXCLUDED_IDENTIFIERS):
        assert f'"{column}"' not in raw


def test_extra_column_closes_the_batch() -> None:
    frame = scoring_frame([example_scoring_row()])
    frame["zip_code"] = "01000"
    batch, events = score_batch(_artifact(), frame, "extra", "contract_probe")
    assert batch["accepted_rows"] == 0
    assert batch["rejected_rows"] == 1
    assert events == []
    assert "zip_code" not in json.dumps(batch)


def test_identical_batch_does_not_alert_and_shift_does() -> None:
    reference, reference_scores = _reference()
    stable, _ = score_batch(
        _artifact(),
        reference,
        "stable",
        "scheduled",
        reference,
        reference_scores,
    )
    assert stable["drift"]["alert"] is False

    shifted = reference.copy()
    shifted["fico_range_low"] = 620.0
    shifted["dti"] = 40.0
    incident, _ = score_batch(
        _artifact(),
        shifted,
        "incident-fico-dti",
        "incident",
        reference,
        reference_scores,
    )
    assert incident["drift"]["alert"] is True
    assert "fico_range_low" in incident["drift"]["alert_columns"]


def test_small_batch_page_is_not_confirmed_by_the_full_sample() -> None:
    from credit_scoring.observability.metrics import confirm_batch_alerts

    batches = [
        {
            "batch_kind": "scheduled",
            "batch_id": "2018-03",
            "drift": {"alert": True, "alert_columns": ["fico_range_low"]},
        }
    ]
    monitor = [
        {
            "label": "referencia contra producao natural",
            "columns": [{"column": "fico_range_low", "psi": 0.02}],
        }
    ]
    confirm_batch_alerts(batches, monitor)
    assert batches[0]["drift"]["confirmed"] is False
    reference, reference_scores = _reference()
    clean, _ = score_batch(
        _artifact(),
        reference.iloc[:20],
        "2016-01",
        "scheduled",
        reference,
        reference_scores,
    )
    dirty_rows = [example_scoring_row(fico_range_low=None), example_scoring_row()]
    dirty, _ = score_batch(_artifact(), scoring_frame(dirty_rows), "probe-contract", "contract_probe")
    summary = summarize_batches([clean, dirty])
    assert summary["prediction_volume_by_kind"]["scheduled"] == 20
    assert summary["error_rate"] > 0
    assert {item["code"] for item in summary["alerts"]} >= {"error_rate", "null_rate"}
    html = render_observability_html(summary, [clean, dirty])
    assert "Taxa de erro" in html
    assert "zip_code" not in html
