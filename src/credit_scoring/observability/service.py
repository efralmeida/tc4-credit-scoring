"""Scoring operacional: contrato, latencia, log e PSI do lote.

O modelo continua sem limiar. Este modulo nao aprova credito. Ele
registra se a inferencia aconteceu, quanto tempo levou e se o lote
ainda parece com a referencia.
"""

import time
from typing import Any

import pandas as pd

from credit_scoring.config import FEATURE_COLUMNS
from credit_scoring.data.schema import SCORING_SCHEMA, quarantine_invalid_rows, summarize_schema_error
from credit_scoring.models.baseline import predict_default_probability
from credit_scoring.observability.logging import new_event_id, score_bucket, utc_now
from credit_scoring.observability.metrics import batch_drift, null_cell_rate

_METADATA_COLUMNS = ("issue_d", "loan_status")


def split_scoring_batch(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aplica o contrato de scoring e nao devolve a linha rejeitada.

    Coluna extra ou coluna faltante fecha o lote inteiro. Falha de
    linha, como DTI 999, tira so a linha. O resumo guarda regra e
    contagem, no mesmo espirito da etapa 1.

    Args:
        frame: Lote recebido. Pode trazer `issue_d` e `loan_status`.

    Returns:
        Linhas aceitas, so com features, e o resumo da rejeicao.
    """
    extra = sorted(
        column
        for column in frame.columns
        if column not in FEATURE_COLUMNS and column not in _METADATA_COLUMNS
    )
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    empty = frame.iloc[0:0].loc[:, []].copy()
    if extra or missing:
        checks = []
        if missing:
            checks.append("coluna obrigatoria ausente")
        if extra:
            checks.append("coluna fora do contrato")
        return empty, {
            "columns": extra or missing,
            "checks": checks,
            "failure_count": len(extra) + len(missing),
            "rejected_rows": int(len(frame)),
            "accepted_rows": 0,
            "message": "Falha de contrato.",
        }
    scoring = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    try:
        return quarantine_invalid_rows(scoring, SCORING_SCHEMA)
    except Exception as error:
        # A segunda validacao da quarentena nao deveria falhar. Se falhar,
        # o lote fecha. O payload continua fora do log.
        summary = summarize_schema_error(error)
        summary["rejected_rows"] = int(len(frame))
        summary["accepted_rows"] = 0
        return empty, summary


def score_batch(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    batch_id: str,
    batch_kind: str,
    reference: pd.DataFrame | None = None,
    reference_scores: pd.Series | None = None,
    scored_at: str | None = None,
    sequence: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pontua um lote e devolve o log agregado e os eventos minimizados.

    Args:
        artifact: Baseline carregado.
        frame: Lote de producao, com ou sem metadado.
        batch_id: Identificador operacional do lote, nao do cliente.
        batch_kind: `scheduled`, `contract_probe` ou `incident`.
        reference: Features da referencia, para o PSI do lote.
        reference_scores: Scores da referencia, alinhados a `reference`.
        scored_at: Instante do lote. Quando ausente, usa o relogio UTC.
        sequence: Ordem de execucao, para o dashboard nao depender do nome.

    Returns:
        Registro do lote e eventos de predicao aceita. Rejeicao nao vira
        evento de linha.
    """
    timestamp = scored_at or utc_now()
    incoming = int(len(frame))
    incoming_null_rate = null_cell_rate(frame)
    accepted, quarantine = split_scoring_batch(frame)
    started = time.perf_counter()
    scores = (
        predict_default_probability(artifact, accepted)
        if len(accepted)
        else pd.Series(dtype=float, name="default_probability")
    )
    elapsed_ms = (time.perf_counter() - started) * 1_000
    per_row = float(elapsed_ms / len(scores)) if len(scores) else None
    drift = {"evaluated": False, "alert": False, "alert_columns": [], "columns": [], "reason": None}
    if reference is not None and reference_scores is not None and len(scores):
        drift = batch_drift(reference, reference_scores, accepted, scores)
    batch = {
        "event_type": "batch",
        "sequence": sequence,
        "batch_id": batch_id,
        "batch_kind": batch_kind,
        "scored_at": timestamp,
        "model_trained_at": _model_stamp(artifact),
        "incoming_rows": incoming,
        "accepted_rows": int(len(scores)),
        "rejected_rows": int(quarantine.get("rejected_rows", 0)),
        "error_rate": float(quarantine.get("rejected_rows", 0) / incoming) if incoming else 0.0,
        "null_cell_rate": incoming_null_rate,
        "latency_ms_batch": float(elapsed_ms),
        "latency_ms_per_row": per_row,
        "score_mean": float(scores.mean()) if len(scores) else None,
        "score_p50": float(scores.quantile(0.5)) if len(scores) else None,
        "score_p95": float(scores.quantile(0.95)) if len(scores) else None,
        "contract_checks": list(quarantine.get("checks", [])),
        "drift": drift,
        "features_logged": False,
        "identifiers_logged": False,
        "rejected_payload_stored": False,
    }
    events = [
        {
            "event_type": "prediction",
            "event_id": new_event_id(),
            "batch_id": batch_id,
            "scored_at": timestamp,
            "status": "accepted",
            "score_bucket": score_bucket(float(score)),
            "features_logged": False,
            "identifiers_logged": False,
        }
        for score in scores.to_numpy()
    ]
    return batch, events


def _model_stamp(artifact: dict[str, Any]) -> str | None:
    """Le a versao do modelo sem abrir dado de treino.

    Args:
        artifact: Baseline com metadados.

    Returns:
        `trained_at`, ou None se o artefato antigo nao tiver o campo.
    """
    metadata = artifact.get("metadata", {})
    stamp = metadata.get("trained_at")
    if stamp is None:
        return None
    return str(stamp)
