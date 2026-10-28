"""Executa a etapa 4: governanca, vies e analise causal.

Nao ha limiar novo. O relatorio explica a base legal do cenario, o que
nao e dado sensivel, o prazo de cada artefato e o que a etapa 2 mediu
sem poder chamar de causa.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_scoring.config import GOVERNANCE_HTML_PATH, GOVERNANCE_REPORT_PATH  # noqa: E402
from credit_scoring.data.prepare import load_reference_dataset  # noqa: E402
from credit_scoring.drift.simulation import (  # noqa: E402
    load_production_dataset,
    production_paths,
    score_applications,
)
from credit_scoring.governance.report import (  # noqa: E402
    build_governance_report,
    write_governance_html,
    write_governance_json,
)
from credit_scoring.models.baseline import load_baseline  # noqa: E402

LOGGER = logging.getLogger(__name__)


def run() -> None:
    """Pontua as tres janelas e grava o relatorio de governanca."""
    reference = load_reference_dataset()
    paths = production_paths()
    natural = load_production_dataset(paths["natural"])
    drifted = load_production_dataset(paths["drifted"])
    artifact = load_baseline()
    report = build_governance_report(
        reference,
        natural,
        drifted,
        score_applications(artifact, reference),
        score_applications(artifact, natural),
        score_applications(artifact, drifted),
    )
    write_governance_json(report, GOVERNANCE_REPORT_PATH)
    write_governance_html(report, GOVERNANCE_HTML_PATH)
    causal = report["causal_analysis"]
    LOGGER.info("Sinais de vies na producao: %s.", len(report["bias"]["natural_production"]["signals"]))
    LOGGER.info(
        "Kitagawa, efeito de taxa: %s.",
        causal["observational_window"]["kitagawa"]["rate_effect"],
    )
    LOGGER.info("Relatorio em %s.", GOVERNANCE_REPORT_PATH)


def main() -> None:
    """Ponto de entrada da etapa 4."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argparse.ArgumentParser(description="Etapa 4 — governanca e conformidade.").parse_args()
    run()


if __name__ == "__main__":
    main()
