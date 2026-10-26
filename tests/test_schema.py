"""Testes dos contratos de scoring e de referencia."""

import pandas as pd
import pytest
from credit_scoring.data.prepare import normalize_application_fields
from credit_scoring.data.schema import (
    CONTRACT_ERRORS,
    REFERENCE_SCHEMA,
    SCORING_SCHEMA,
    example_scoring_row,
    quarantine_invalid_rows,
    reference_frame,
    scoring_frame,
)
from credit_scoring.validation.failure_demo import (
    assert_failure_report,
    demonstrate_controlled_failure,
)


def test_valid_scoring_frame_passes() -> None:
    validated = SCORING_SCHEMA.validate(scoring_frame())
    assert list(validated.columns) == list(scoring_frame().columns)
    assert validated.loc[0, "annual_inc"] == 65_000.0


def test_reference_window_rejects_future_origination() -> None:
    frame = reference_frame([{"issue_d": pd.Timestamp("2017-03-01")}])
    with pytest.raises(CONTRACT_ERRORS):
        REFERENCE_SCHEMA.validate(frame, lazy=True)


def test_quarantine_drops_only_invalid_rows() -> None:
    frame = reference_frame(
        [
            {},
            {"dti": 999.0},
            {"annual_inc": 0.0},
        ]
    )
    accepted, summary = quarantine_invalid_rows(frame, REFERENCE_SCHEMA)
    assert len(accepted) == 1
    assert summary["rejected_rows"] == 2
    assert "dti" in summary["columns"]
    assert "annual_inc" in summary["columns"]


def test_missing_employment_length_becomes_known_category() -> None:
    frame = scoring_frame([example_scoring_row(emp_length=pd.NA)])
    normalized = normalize_application_fields(frame)
    validated = SCORING_SCHEMA.validate(normalized)
    assert validated.loc[0, "emp_length"] == "unknown"


def test_controlled_failure_rejects_every_invalid_case() -> None:
    report = demonstrate_controlled_failure()
    assert_failure_report(report)
    assert report["control_valid_passed"] is True
    assert report["invalid_cases_rejected"] == 6


def test_strict_schema_blocks_identifier_column() -> None:
    frame = scoring_frame().assign(zip_code="01310")
    with pytest.raises(CONTRACT_ERRORS):
        SCORING_SCHEMA.validate(frame, lazy=True)
