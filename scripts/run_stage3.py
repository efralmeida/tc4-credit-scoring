"""Executa a etapa 3: logs de predicao, saude e drift operacional.

O fluxo trata a producao natural como lotes mensais. Dois lotes extras
nao sao clientes novos:

- contract_probe: linhas validas adulteradas, para a taxa de erro nao
  ficar artificialmente em zero.
- incident: recorte da producao com FICO e DTI ja deslocados, para o
  PSI de lote disparar.

O PSI da amostra inteira confirma o lote. Ele usa a mesma regra da
etapa 2. O payload rejeitado nao e gravado.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_scoring.config import (  # noqa: E402
    BATCH_HEALTH_LOG_PATH,
    CONTRACT_PROBE_ROWS,
    FEATURE_COLUMNS,
    HEALTH_SUMMARY_PATH,
    INCIDENT_BATCH_ROWS,
    OBSERVABILITY_REPORT_PATH,
    PREDICTION_LOG_PATH,
    PREDICTION_LOG_SAMPLE_PATH,
    PREDICTION_LOG_SAMPLE_SIZE,
    RANDOM_STATE,
)
from credit_scoring.data.prepare import load_reference_dataset  # noqa: E402
from credit_scoring.drift.simulation import (  # noqa: E402
    load_production_dataset,
    production_paths,
    score_applications,
)
from credit_scoring.models.baseline import load_baseline  # noqa: E402
from credit_scoring.observability.dashboard import write_observability_html  # noqa: E402
from credit_scoring.observability.logging import utc_now, write_jsonl  # noqa: E402
from credit_scoring.observability.metrics import (  # noqa: E402
    authoritative_drift,
    confirm_batch_alerts,
    summarize_batches,
)
from credit_scoring.observability.service import score_batch  # noqa: E402

LOGGER = logging.getLogger(__name__)


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Grava o resumo de saude sem dados de linha.

    Args:
        payload: Conteudo serializavel.
        path: Destino.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def monthly_batches(natural: Any) -> list[tuple[str, Any]]:
    """Agrupa a producao natural pelo mes de originacao.

    Args:
        natural: Producao validada, com `issue_d`.

    Returns:
        Pares de id do lote e fatia, em ordem cronologica.
    """
    dated = natural.copy()
    dated["_month"] = dated["issue_d"].dt.to_period("M").astype(str)
    return [
        (str(month), frame.drop(columns="_month"))
        for month, frame in dated.groupby("_month", sort=True)
    ]


def contract_probe(natural: Any) -> Any:
    """Adultera um recorte pequeno para o log registrar rejeicao.

    Args:
        natural: Producao valida.

    Returns:
        Lote com nulo e DTI fora do contrato. Nao deve ser persistido.
    """
    probe = natural.iloc[:CONTRACT_PROBE_ROWS].loc[:, list(FEATURE_COLUMNS)].copy()
    probe.iloc[:8, probe.columns.get_loc("fico_range_low")] = None
    probe.iloc[8:12, probe.columns.get_loc("dti")] = 999.0
    return probe


def incident_batch(drifted: Any) -> Any:
    """Recorta a producao deslocada para simular uma janela ruim.

    Args:
        drifted: Producao com FICO e DTI alterados.

    Returns:
        Amostra do incidente, ainda com metadados.
    """
    size = min(INCIDENT_BATCH_ROWS, len(drifted))
    return drifted.sample(n=size, random_state=RANDOM_STATE)


def run() -> dict[str, Any]:
    """Pontua os lotes, grava logs e o HTML.

    Returns:
        Resumo de saude.
    """
    reference = load_reference_dataset()
    paths = production_paths()
    natural = load_production_dataset(paths["natural"])
    drifted = load_production_dataset(paths["drifted"])
    artifact = load_baseline()
    reference_scores = score_applications(artifact, reference)
    natural_scores = score_applications(artifact, natural)
    drifted_scores = score_applications(artifact, drifted)
    timestamp = utc_now()
    batches: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    sequence = 1
    jobs: list[tuple[str, str, Any]] = [
        (batch_id, "scheduled", frame) for batch_id, frame in monthly_batches(natural)
    ]
    jobs.append(("probe-contract", "contract_probe", contract_probe(natural)))
    jobs.append(("incident-fico-dti", "incident", incident_batch(drifted)))
    for batch_id, kind, frame in jobs:
        batch, batch_events = score_batch(
            artifact,
            frame,
            batch_id,
            kind,
            reference.loc[:, list(FEATURE_COLUMNS)],
            reference_scores,
            scored_at=timestamp,
            sequence=sequence,
        )
        batches.append(batch)
        events.extend(batch_events)
        sequence += 1
        LOGGER.info(
            "Lote %s: aceitas=%s rejeitadas=%s alerta=%s.",
            batch_id,
            batch["accepted_rows"],
            batch["rejected_rows"],
            batch["drift"]["alert"],
        )
    summary = summarize_batches(batches)
    summary["generated_at"] = timestamp
    summary["model_trained_at"] = artifact.get("metadata", {}).get("trained_at")
    summary["drift_monitor"] = [
        authoritative_drift(
            reference,
            natural,
            reference_scores,
            natural_scores,
            "referencia contra producao natural",
        ),
        authoritative_drift(
            reference,
            drifted,
            reference_scores,
            drifted_scores,
            "referencia contra producao deslocada",
        ),
    ]
    confirm_batch_alerts(batches, summary["drift_monitor"])
    summary["drift_alert_count"] = sum(1 for batch in batches if batch["drift"]["alert"])
    summary["drift_alert_batches"] = [
        batch["batch_id"] for batch in batches if batch["drift"]["alert"]
    ]
    summary["confirmed_drift_batches"] = [
        batch["batch_id"] for batch in batches if batch["drift"].get("confirmed")
    ]
    summary["privacy"] = {
        "features_logged": False,
        "identifiers_logged": False,
        "rejected_payload_stored": False,
        "prediction_log_stores_score_bucket_only": True,
        "operational_volume_includes_probe_and_incident": True,
    }
    summary["evidence"] = {
        "null_cell_rate": "Percentual de celulas vazias nas colunas do contrato, antes da rejeicao.",
        "latency": "Tempo de predict_proba do lote aceito, dividido pelo numero de linhas.",
        "error_rate": "Linhas rejeitadas sobre linhas recebidas. O payload nao e gravado.",
        "drift": (
            "Alerta de lote quando o PSI de FICO, DTI ou score passa de 0,25 "
            "e o lote tem volume minimo. A confirmacao e o PSI da amostra inteira. "
            "Nao e p-valor e nao usa o KS."
        ),
    }
    write_jsonl(BATCH_HEALTH_LOG_PATH, batches)
    write_jsonl(PREDICTION_LOG_PATH, events)
    write_jsonl(PREDICTION_LOG_SAMPLE_PATH, events[:PREDICTION_LOG_SAMPLE_SIZE])
    write_json(summary, HEALTH_SUMMARY_PATH)
    write_observability_html(summary, batches, OBSERVABILITY_REPORT_PATH)
    LOGGER.info("Resumo em %s.", HEALTH_SUMMARY_PATH)
    LOGGER.info("Dashboard em %s.", OBSERVABILITY_REPORT_PATH)
    return summary


def main() -> None:
    """Ponto de entrada da etapa 3."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argparse.ArgumentParser(description="Etapa 3 — observabilidade do credit scoring.").parse_args()
    run()


if __name__ == "__main__":
    main()
