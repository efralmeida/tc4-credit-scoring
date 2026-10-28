"""Indicadores de saude do scoring e leitura operacional do drift.

Cinco perguntas, todas em agregado:

- Quanto do lote chegou vazio?
- Quanto tempo a inferencia levou?
- Quantas predicoes foram aceitas?
- Quantas linhas o contrato barrou?
- O PSI do lote, ou da amostra inteira, saiu da faixa estavel?

O PSI desta etapa e o mesmo da etapa 2. O alerta de lote usa a faixa
relevante, 0,25, porque lote mensal e pequeno e o PSI moderado oscila.
O KS nao pagina. Ele continua sendo apoio, no relatorio da etapa 2.
"""

from typing import Any

import pandas as pd

from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    LATENCY_MS_PER_ROW_ALERT,
    MIN_BATCH_DRIFT_ROWS,
    NUMERIC_FEATURES,
    PSI_SIGNIFICANT_THRESHOLD,
)
from credit_scoring.drift.statistics import (
    compare_columns,
    population_stability_index,
    psi_band,
)


def null_cell_rate(frame: pd.DataFrame) -> float:
    """Mede o percentual de celulas vazias nas colunas do contrato.

    Colunas extras nao entram. O indicador olha a entrada, antes da
    rejeicao, porque depois do contrato o nulo ja foi embora.

    Args:
        frame: Lote recebido pelo scoring, ainda nao filtrado.

    Returns:
        Proporcao de nulos entre as features presentes. Lote sem
        feature de contrato vale 1.
    """
    present = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if frame.empty or not present:
        return 1.0
    return float(frame.loc[:, present].isna().to_numpy().mean())


def batch_drift(
    reference: pd.DataFrame,
    reference_scores: pd.Series,
    current: pd.DataFrame,
    current_scores: pd.Series,
) -> dict[str, Any]:
    """Compara FICO, DTI e score do lote com a referencia.

    Args:
        reference: Features da referencia.
        reference_scores: Scores da referencia.
        current: Features aceitas do lote.
        current_scores: Scores do lote.

    Returns:
        PSI das tres series e a decisao de alerta. Lote curto nao alerta.
    """
    if len(current_scores) < MIN_BATCH_DRIFT_ROWS:
        return {
            "evaluated": False,
            "reason": "lote abaixo do minimo para PSI",
            "min_rows": MIN_BATCH_DRIFT_ROWS,
            "alert": False,
            "alert_columns": [],
            "columns": [],
        }
    measured = {
        "fico_range_low": population_stability_index(
            reference["fico_range_low"],
            current["fico_range_low"],
        ),
        "dti": population_stability_index(reference["dti"], current["dti"]),
        "default_probability": population_stability_index(reference_scores, current_scores),
    }
    columns = [
        {"column": name, "psi": value, "psi_band": psi_band(value)}
        for name, value in measured.items()
    ]
    alert_columns = [
        item["column"] for item in columns if item["psi"] >= PSI_SIGNIFICANT_THRESHOLD
    ]
    return {
        "evaluated": True,
        "reason": None,
        "min_rows": MIN_BATCH_DRIFT_ROWS,
        "alert": bool(alert_columns),
        "alert_columns": alert_columns,
        "columns": columns,
    }


def authoritative_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_scores: pd.Series,
    current_scores: pd.Series,
    label: str,
) -> dict[str, Any]:
    """Roda o PSI da amostra inteira, que confirma o alerta de lote.

    Args:
        reference: Referencia validada.
        current: Producao natural ou deslocada.
        reference_scores: Scores da referencia.
        current_scores: Scores da producao.
        label: Nome da comparacao, para o relatorio.

    Returns:
        Colunas com PSI moderado ou maior, mais o PSI do score.
    """
    reference_monitor = reference.loc[:, list(FEATURE_COLUMNS)].copy()
    current_monitor = current.loc[:, list(FEATURE_COLUMNS)].copy()
    reference_monitor["default_probability"] = reference_scores.to_numpy()
    current_monitor["default_probability"] = current_scores.to_numpy()
    columns = compare_columns(
        reference_monitor,
        current_monitor,
        [*NUMERIC_FEATURES, "default_probability"],
        list(CATEGORICAL_FEATURES),
    )
    affected = [row for row in columns if row["affected"]]
    score = next(row for row in columns if row["column"] == "default_probability")
    return {
        "label": label,
        "evidence": (
            "PSI na amostra inteira. Abaixo de 0,10 estavel; de 0,10 a 0,25 "
            "moderado; a partir de 0,25 relevante. Nao e p-valor. O KS nao "
            "entra neste alerta."
        ),
        "score_psi": score["psi"],
        "score_psi_band": score["psi_band"],
        "affected_columns": [row["column"] for row in affected],
        "columns": columns,
    }


def confirm_batch_alerts(
    batches: list[dict[str, Any]],
    drift_monitor: list[dict[str, Any]],
) -> None:
    """Marca se o PSI do lote continua relevante na amostra inteira.

    Lote de poucas centenas de linhas pagina com facilidade. O alerta so
    fica confirmado quando a mesma coluna passa de 0,25 na comparacao
    completa correspondente.

    Args:
        batches: Lotes ja pontuados. A funcao altera o bloco `drift`.
        drift_monitor: Comparacoes de amostra inteira, na ordem natural
            e depois deslocada.
    """
    confirmed_columns = {
        "scheduled": _significant_columns(_monitor_by_label(drift_monitor, "natural")),
        "incident": _significant_columns(_monitor_by_label(drift_monitor, "deslocada")),
        "contract_probe": set(),
    }
    for batch in batches:
        kind = str(batch["batch_kind"])
        expected = confirmed_columns.get(kind, set())
        alert_columns = list(batch["drift"].get("alert_columns", []))
        confirmed = sorted(column for column in alert_columns if column in expected)
        batch["drift"]["confirmed_columns"] = confirmed
        batch["drift"]["confirmed"] = bool(confirmed)


def _monitor_by_label(drift_monitor: list[dict[str, Any]], fragment: str) -> dict[str, Any]:
    """Acha a comparacao cujo rotulo contem o fragmento.

    Args:
        drift_monitor: Blocos de PSI da amostra inteira.
        fragment: Trecho do rotulo, como `natural` ou `deslocada`.

    Returns:
        Primeiro bloco compativel, ou um bloco vazio.
    """
    for item in drift_monitor:
        if fragment in str(item.get("label", "")):
            return item
    return {"columns": []}


def _significant_columns(item: dict[str, Any]) -> set[str]:
    """Lista colunas com PSI na faixa relevante.

    Args:
        item: Bloco de uma comparacao.

    Returns:
        Nomes das colunas com PSI a partir de 0,25.
    """
    return {
        str(row["column"])
        for row in item.get("columns", [])
        if float(row["psi"]) >= PSI_SIGNIFICANT_THRESHOLD
    }


def summarize_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega os lotes nos cinco indicadores de saude.

    Args:
        batches: Registros gravados no log de lote.

    Returns:
        Volume, nulos, latencia, erro, alertas de drift e a lista de
        alertas que o HTML mostra.
    """
    incoming = sum(int(batch["incoming_rows"]) for batch in batches)
    accepted = sum(int(batch["accepted_rows"]) for batch in batches)
    rejected = sum(int(batch["rejected_rows"]) for batch in batches)
    null_weight = sum(float(batch["null_cell_rate"]) * int(batch["incoming_rows"]) for batch in batches)
    latency_weight = 0.0
    latency_rows = 0
    latency_values: list[float] = []
    drift_alerts = [batch["batch_id"] for batch in batches if batch["drift"]["alert"]]
    for batch in batches:
        per_row = batch.get("latency_ms_per_row")
        if per_row is None or int(batch["accepted_rows"]) == 0:
            continue
        latency_weight += float(per_row) * int(batch["accepted_rows"])
        latency_rows += int(batch["accepted_rows"])
        latency_values.append(float(per_row))
    mean_latency = float(latency_weight / latency_rows) if latency_rows else None
    p95_latency = _percentile(latency_values, 0.95)
    error_rate = float(rejected / incoming) if incoming else 0.0
    null_rate = float(null_weight / incoming) if incoming else 0.0
    alerts = _health_alerts(null_rate, error_rate, mean_latency, drift_alerts)
    by_kind = {
        kind: sum(int(batch["accepted_rows"]) for batch in batches if batch["batch_kind"] == kind)
        for kind in ("scheduled", "contract_probe", "incident")
    }
    return {
        "batch_count": len(batches),
        "incoming_volume": incoming,
        "prediction_volume": accepted,
        "prediction_volume_by_kind": by_kind,
        "rejected_volume": rejected,
        "null_cell_rate": null_rate,
        "mean_latency_ms": mean_latency,
        "p95_batch_latency_ms": p95_latency,
        "error_rate": error_rate,
        "drift_alert_count": len(drift_alerts),
        "drift_alert_batches": drift_alerts,
        "thresholds": {
            "null_cell_rate": 0.0,
            "error_rate": 0.0,
            "latency_ms_per_row": LATENCY_MS_PER_ROW_ALERT,
            "batch_psi": PSI_SIGNIFICANT_THRESHOLD,
            "min_batch_drift_rows": MIN_BATCH_DRIFT_ROWS,
        },
        "alerts": alerts,
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    """Calcula um quantil simples sem depender do tamanho do lote.

    Args:
        values: Latencias por lote.
        quantile: Quantil entre 0 e 1.

    Returns:
        Valor do quantil, ou None se nao houve inferencia.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
    return float(ordered[index])


def _health_alerts(
    null_rate: float,
    error_rate: float,
    mean_latency: float | None,
    drift_alerts: list[str],
) -> list[dict[str, str]]:
    """Traduz indicador fora da regra em alerta operacional.

    Qualquer rejeicao de contrato pagina. Neste desenho o contrato e a
    porta de entrada: uma linha invalida nao deve ser tratada como ruido.

    Args:
        null_rate: Taxa de nulos ponderada pelo volume.
        error_rate: Rejeitadas sobre recebidas.
        mean_latency: Latencia media por linha, em milissegundos.
        drift_alerts: Ids dos lotes com PSI relevante.

    Returns:
        Alertas com codigo e mensagem, sem dado de cliente.
    """
    alerts: list[dict[str, str]] = []
    if null_rate > 0:
        alerts.append(
            {
                "code": "null_rate",
                "message": "Ha celulas nulas nas colunas do contrato antes da rejeicao.",
            }
        )
    if error_rate > 0:
        alerts.append(
            {
                "code": "error_rate",
                "message": "O contrato rejeitou pelo menos uma linha. O payload nao foi gravado.",
            }
        )
    if mean_latency is not None and mean_latency > LATENCY_MS_PER_ROW_ALERT:
        alerts.append(
            {
                "code": "latency",
                "message": "A latencia media por linha passou do limiar operacional.",
            }
        )
    if drift_alerts:
        alerts.append(
            {
                "code": "drift",
                "message": "PSI relevante em FICO, DTI ou score. Confirmar na amostra inteira.",
            }
        )
    return alerts
