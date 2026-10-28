"""Junta governanca num JSON e num HTML que da para abrir no video.

O README continua sendo o texto de entrega. O HTML repete o suficiente
para mostrar catalogo, prazo, sinal de vies e a decomposicao, sem linha
de cliente.
"""

import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from credit_scoring.governance.bias import audit_slices
from credit_scoring.governance.catalog import privacy_principles
from credit_scoring.governance.causal import build_causal_analysis
from credit_scoring.governance.retention import retention_manifest


def build_governance_report(
    reference: pd.DataFrame,
    natural: pd.DataFrame,
    drifted: pd.DataFrame,
    reference_scores: pd.Series,
    natural_scores: pd.Series,
    drifted_scores: pd.Series,
) -> dict[str, Any]:
    """Monta o relatorio da etapa 4.

    Args:
        reference: Referencia validada.
        natural: Producao natural.
        drifted: Producao com a intervencao.
        reference_scores: Scores da referencia.
        natural_scores: Scores da producao natural.
        drifted_scores: Scores da producao deslocada.

    Returns:
        Governanca, vies e analise causal, sem dado de linha.
    """
    return {
        "privacy": privacy_principles(),
        "retention": retention_manifest(),
        "bias": {
            "reference": audit_slices(reference, reference_scores, _target(reference)),
            "natural_production": audit_slices(natural, natural_scores, _target(natural)),
        },
        "causal_analysis": build_causal_analysis(
            reference,
            natural,
            drifted,
            reference_scores,
            natural_scores,
            drifted_scores,
        ),
    }


def write_governance_json(report: dict[str, Any], path: Path) -> None:
    """Grava o JSON da etapa 4.

    Args:
        report: Saida de `build_governance_report`.
        path: Destino.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def render_governance_html(report: dict[str, Any]) -> str:
    """Renderiza a pagina de governanca.

    Args:
        report: Relatorio ja agregado.

    Returns:
        HTML autonomo.
    """
    privacy = report["privacy"]
    causal = report["causal_analysis"]
    window = causal["observational_window"]
    kitagawa = window["kitagawa"]
    intervention = causal["intervention"]
    signals = report["bias"]["natural_production"]["signals"]
    signal_html = (
        "".join(
            "<li>"
            f"{escape(item['column'])} / {escape(item['group'])}: "
            f"impacto desproporcional {item['disparate_impact']:.2f} "
            f"contra {escape(item['reference_group'])}."
            "</li>"
            for item in signals
        )
        or "<li>Nenhum recorte da producao natural ficou abaixo de quatro quintos, no volume minimo.</li>"
    )
    catalog_rows = "\n".join(
        "<tr>"
        f"<td>{escape(item['column'])}</td>"
        f"<td>{escape(item['lgpd'])}</td>"
        f"<td>{escape(item['sensitive_personal_data'])}</td>"
        f"<td>{escape(item['decision'])}</td>"
        "</tr>"
        for item in privacy["catalog"]
        if item["decision"] != "entra no baseline"
    )
    retention_rows = "\n".join(
        "<tr>"
        f"<td>{escape(item['artifact'])}</td>"
        f"<td>{escape(item['git'])}</td>"
        f"<td>{escape(item['scenario_retention'])}</td>"
        "</tr>"
        for item in report["retention"]["schedule"]
    )
    bases = "".join(
        f"<li><strong>{escape(item['purpose'])}.</strong> {escape(item['basis'])} "
        f"{escape(item['limit'])}</li>"
        for item in privacy["legal_bases"]
    )
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>Governanca e LGPD do credit scoring</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 980px; color: #1c1917; background: #faf7f2; }}
    table {{ width: 100%; border-collapse: collapse; background: white; font-size: 0.92rem; }}
    th, td {{ border-bottom: 1px solid #e7e5e4; text-align: left; padding: 0.35rem 0.4rem; vertical-align: top; }}
    .note {{ color: #44403c; }}
  </style>
</head>
<body>
  <h1>Governanca, LGPD e analise causal</h1>
  <p class="note">{escape(privacy["scenario"])}</p>
  <h2>Bases legais</h2>
  <ul>{bases}</ul>
  <p>{escape(privacy["automated_decision"]["current_behavior"])}</p>
  <h2>O que nao entra</h2>
  <table>
    <thead><tr><th>Coluna</th><th>Classe</th><th>Dado sensivel</th><th>Decisao</th></tr></thead>
    <tbody>{catalog_rows}</tbody>
  </table>
  <h2>Retencao</h2>
  <table>
    <thead><tr><th>Artefato</th><th>Git</th><th>Prazo no cenario</th></tr></thead>
    <tbody>{retention_rows}</tbody>
  </table>
  <h2>Vies</h2>
  <p class="note">O modelo nao usa atributo protegido. O sinal abaixo compara massa de score. Nao e concessao e nao e laudo de discriminacao.</p>
  <ul>{signal_html}</ul>
  <h2>Analise causal</h2>
  <p>Intervencao: score medio de {intervention["score_mean_before"]:.3f} para {intervention["score_mean_after"]:.3f}. PSI do score {intervention["score_psi"]:.3f}. Variacao do rotulo {intervention["label_rate_delta"]:.3f}. {escape(intervention["claim"])}</p>
  <p>Janela real: default de {window["default_rate_reference"]:.3f} para {window["default_rate_natural"]:.3f}. Efeito de composicao {kitagawa["composition_effect"]:.3f}. Efeito de taxa {kitagawa["rate_effect"]:.3f}. Residuo {kitagawa["residual"]:.3f}. {escape(window["claim"])}</p>
  <p class="note">{escape(kitagawa["reading"])}</p>
</body>
</html>
"""


def write_governance_html(report: dict[str, Any], path: Path) -> None:
    """Grava o HTML da etapa 4.

    Args:
        report: Relatorio agregado.
        path: Destino.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_governance_html(report), encoding="utf-8")


def _target(frame: pd.DataFrame) -> pd.Series:
    """Converte o desfecho sem importar o modulo de modelo no topo.

    Args:
        frame: Referencia ou producao.

    Returns:
        Serie binaria de Charged Off.
    """
    from credit_scoring.models.baseline import defaulted_target

    return defaulted_target(frame["loan_status"])
