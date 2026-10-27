"""Executa a etapa 2: producao, drift intencional, scores e relatorio.

O fluxo tem quatro perguntas, nesta ordem:

1. Os dados de producao passam no contrato? Se nao passam, a etapa 1
   barra o lote. Drift so e medido em dado aceito.
2. A janela 2016-2018, sem mexida artificial, ja difere de 2012-2015?
3. Depois de piorar FICO e DTI de proposito, o PSI aponta essas colunas?
4. O score do baseline se move, e a taxa de default na mesma faixa de
   score continua parecida?

O HTML principal compara a referencia com a producao deslocada. O
segundo HTML compara a producao natural com a deslocada, para separar
passagem do tempo e alteracao intencional.
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
    DRIFT_SUMMARY_PATH,
    PRODUCTION_SIMULATION_PATH,
)
from credit_scoring.data.prepare import load_reference_dataset  # noqa: E402
from credit_scoring.drift.reports import write_stage2_reports  # noqa: E402
from credit_scoring.drift.simulation import (  # noqa: E402
    apply_intentional_drift,
    load_production_dataset,
    observed_default_rate,
    prediction_summary,
    prepare_production_from_raw,
    prepare_production_from_reference,
    production_paths,
    save_production_dataset,
    score_applications,
)
from credit_scoring.models.baseline import load_baseline  # noqa: E402

LOGGER = logging.getLogger(__name__)


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Grava o resumo da etapa 2 sem dados de linha.

    Args:
        payload: Conteudo serializavel.
        path: Destino do relatorio.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def simulate(from_reference: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cria a producao natural e a producao com FICO e DTI deslocados.

    Args:
        from_reference: Quando verdadeiro, nao relê o arquivo bruto.

    Returns:
        Relatorio da origem e relatorio da injecao.
    """
    reference = load_reference_dataset()
    if from_reference:
        natural, simulation_report = prepare_production_from_reference(reference)
    else:
        natural, simulation_report = prepare_production_from_raw()
    drifted, injection_report = apply_intentional_drift(natural)
    paths = production_paths()
    save_production_dataset(natural, paths["natural"])
    save_production_dataset(drifted, paths["drifted"])
    write_json(
        {"simulation": simulation_report, "injection": injection_report},
        PRODUCTION_SIMULATION_PATH,
    )
    LOGGER.info("Producao natural em %s.", paths["natural"])
    LOGGER.info("Producao deslocada em %s.", paths["drifted"])
    LOGGER.info("Colunas alteradas de proposito: %s.", injection_report["intentional_columns"])
    return simulation_report, injection_report


def load_saved_simulation() -> tuple[dict[str, Any], dict[str, Any]]:
    """Recupera a origem da producao quando `detect` roda sozinho.

    Returns:
        Relatorio da simulacao e relatorio da injecao.

    Raises:
        FileNotFoundError: Se `simulate` ainda nao gravou o estado.
    """
    if not PRODUCTION_SIMULATION_PATH.exists():
        raise FileNotFoundError(
            f"Estado da simulacao ausente em {PRODUCTION_SIMULATION_PATH}. "
            "Execute simulate antes de detect."
        )
    payload = json.loads(PRODUCTION_SIMULATION_PATH.read_text(encoding="utf-8"))
    return payload["simulation"], payload["injection"]


def detect(simulation_report: dict[str, Any], injection_report: dict[str, Any]) -> None:
    """Pontua os tres conjuntos e grava HTML e JSON.

    Args:
        simulation_report: Origem da producao natural.
        injection_report: Transformacoes de FICO e DTI.
    """
    reference = load_reference_dataset()
    paths = production_paths()
    natural = load_production_dataset(paths["natural"])
    drifted = load_production_dataset(paths["drifted"])
    artifact = load_baseline()
    reference_scores = score_applications(artifact, reference)
    natural_scores = score_applications(artifact, natural)
    drifted_scores = score_applications(artifact, drifted)
    summary = write_stage2_reports(
        reference,
        natural,
        drifted,
        reference_scores,
        natural_scores,
        drifted_scores,
        simulation_report,
        injection_report,
    )
    summary["scores"] = {
        "reference": prediction_summary(reference_scores),
        "natural": prediction_summary(natural_scores),
        "drifted": prediction_summary(drifted_scores),
    }
    summary["observed_default_rate"] = {
        "reference": observed_default_rate(reference),
        "natural": observed_default_rate(natural),
        "drifted": observed_default_rate(drifted),
    }
    write_json(summary, DRIFT_SUMMARY_PATH)
    LOGGER.info("Variaveis apontadas pela injecao: %s.", summary["affected_by_injection"])
    LOGGER.info("PSI do score, referencia contra producao deslocada: %s.", summary["prediction_drift"])
    LOGGER.info("Resumo em %s.", DRIFT_SUMMARY_PATH)


def parse_args() -> argparse.Namespace:
    """Le o comando da etapa 2.

    Returns:
        Argumentos de linha de comando.
    """
    parser = argparse.ArgumentParser(description="Etapa 2 — simulacao e deteccao de drift.")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "simulate", "detect"),
        help="Etapa a executar. O padrao simula e depois gera o relatorio.",
    )
    parser.add_argument(
        "--from-reference",
        action="store_true",
        help="Nao relê o bruto. Reposiciona a referencia em 2016-2018 so para demonstracao.",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do fluxo da etapa 2."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.command in {"all", "simulate"}:
        simulation_report, injection_report = simulate(args.from_reference)
    else:
        simulation_report, injection_report = load_saved_simulation()
    if args.command in {"all", "detect"}:
        detect(simulation_report, injection_report)


if __name__ == "__main__":
    main()
