"""Executa a etapa 1: contrato, baseline e falha controlada."""

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
    COEFFICIENTS_PATH,
    FAILURE_REPORT_PATH,
    METRICS_PATH,
    PREPARATION_REPORT_PATH,
)
from credit_scoring.data.prepare import (  # noqa: E402
    load_reference_dataset,
    prepare_reference_dataset,
    save_reference_dataset,
)
from credit_scoring.models.baseline import (  # noqa: E402
    predict_default_probability,
    save_baseline,
    save_coefficients,
    train_baseline,
)
from credit_scoring.validation.failure_demo import (  # noqa: E402
    assert_failure_report,
    demonstrate_controlled_failure,
)

LOGGER = logging.getLogger(__name__)


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Grava um relatorio JSON sem dados de linha.

    Args:
        payload: Conteudo serializavel.
        path: Destino do relatorio.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_prepare() -> None:
    """Gera a amostra de referencia e o relatorio de quarentena."""
    reference, report = prepare_reference_dataset()
    save_reference_dataset(reference)
    write_json(report, PREPARATION_REPORT_PATH)
    LOGGER.info(
        "Referencia pronta: %s linhas. Relatorio em %s.",
        report["rows_reference_sample"],
        PREPARATION_REPORT_PATH,
    )


def run_train() -> None:
    """Treina o baseline sobre a referencia ja contratada."""
    reference = load_reference_dataset()
    artifact = train_baseline(reference)
    save_baseline(artifact)
    save_coefficients(artifact["coefficients"])
    write_json(
        {"metrics": artifact["metrics"], "metadata": artifact["metadata"]},
        METRICS_PATH,
    )
    probe = reference.drop(columns=["issue_d", "loan_status"]).head(5)
    scores = predict_default_probability(artifact, probe)
    LOGGER.info("Metricas: %s", artifact["metrics"])
    LOGGER.info("Amostra de probabilidades apos o contrato: %s", scores.round(4).tolist())
    LOGGER.info("Coeficientes em %s.", COEFFICIENTS_PATH)


def run_failure_demo() -> None:
    """Prova que payloads incorretos sao rejeitados antes da inferencia."""
    report = demonstrate_controlled_failure()
    assert_failure_report(report)
    write_json(report, FAILURE_REPORT_PATH)
    LOGGER.info(
        "Falha controlada confirmada em %s casos. Relatorio em %s.",
        report["invalid_cases_rejected"],
        FAILURE_REPORT_PATH,
    )


def parse_args() -> argparse.Namespace:
    """Le o comando da etapa 1.

    Returns:
        Argumentos de linha de comando.
    """
    parser = argparse.ArgumentParser(description="Etapa 1 — validacao e baseline.")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "prepare", "train", "demonstrate-failure"),
        help="Etapa a executar. O padrao executa o fluxo completo.",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do fluxo da etapa 1."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.command in {"all", "prepare"}:
        run_prepare()
    if args.command in {"all", "train"}:
        run_train()
    if args.command in {"all", "demonstrate-failure"}:
        run_failure_demo()


if __name__ == "__main__":
    main()
