"""Relatorio HTML da saude operacional.

O arquivo e autonomo: nao chama CDN e nao embute linha de cliente.
Serve como evidencia visual da etapa 3 quando nao ha Grafana.
"""

from html import escape
from pathlib import Path
from typing import Any


def render_observability_html(summary: dict[str, Any], batches: list[dict[str, Any]]) -> str:
    """Monta o dashboard estatico a partir do resumo e dos lotes.

    Args:
        summary: Saida de `summarize_batches`, ja com o drift confirmatorio.
        batches: Log de lote, na ordem de execucao.

    Returns:
        HTML completo.
    """
    cards = "".join(
        _card(label, value)
        for label, value in (
            ("Predicoes aceitas", _integer(summary["prediction_volume"])),
            ("Taxa de erro", _percent(summary["error_rate"])),
            ("Celulas nulas", _percent(summary["null_cell_rate"])),
            ("Latencia media", _latency(summary["mean_latency_ms"])),
            ("Alertas de drift", str(summary["drift_alert_count"])),
        )
    )
    alerts = summary.get("alerts") or []
    alert_html = (
        "".join(f"<li><strong>{escape(item['code'])}.</strong> {escape(item['message'])}</li>" for item in alerts)
        if alerts
        else "<li>Nenhum alerta nas regras configuradas.</li>"
    )
    rows = "\n".join(_batch_row(batch) for batch in batches)
    drift_blocks = "\n".join(
        _drift_table(item) for item in summary.get("drift_monitor", [])
    )
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>Observabilidade do credit scoring</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 1080px; color: #1c1917; background: #faf7f2; }}
    h1, h2 {{ font-weight: 600; }}
    .cards {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.75rem; }}
    .card {{ background: white; border: 1px solid #e7e5e4; padding: 0.8rem; }}
    .card span {{ display: block; font-size: 0.8rem; color: #57534e; }}
    .card strong {{ font-size: 1.3rem; }}
    table {{ width: 100%; border-collapse: collapse; background: white; font-size: 0.92rem; }}
    th, td {{ border-bottom: 1px solid #e7e5e4; text-align: left; padding: 0.35rem 0.4rem; }}
    .alert {{ background: #fff7ed; border-left: 4px solid #c2410c; padding: 0.6rem 1rem; }}
    .note {{ color: #44403c; font-size: 0.95rem; }}
  </style>
</head>
<body>
  <h1>Saude operacional do modelo</h1>
  <p class="note">Lotes de scoring sem identificador e sem payload rejeitado. O limiar de concessao continua fora do baseline. PSI abaixo de 0,10 e estavel; a partir de 0,25 o lote pagina. O KS nao e a regra de alerta.</p>
  <section class="cards">{cards}</section>
  <h2>Alertas</h2>
  <div class="alert"><ul>{alert_html}</ul></div>
  <h2>Lotes</h2>
  <table>
    <thead>
      <tr>
        <th>Lote</th><th>Tipo</th><th>Aceitas</th><th>Rejeitadas</th>
        <th>Nulos</th><th>ms/linha</th><th>Score medio</th><th>Drift</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  <h2>Drift confirmatorio, amostra inteira</h2>
  <p class="note">O PSI do lote mensal oscila. A confirmacao usa a mesma definicao da etapa 2, na amostra completa, nao um p-valor.</p>
  {drift_blocks}
  <p class="note">Gerado em {escape(str(summary.get("generated_at", "")))}. Features e identificadores nao foram gravados neste relatorio.</p>
</body>
</html>
"""


def write_observability_html(
    summary: dict[str, Any],
    batches: list[dict[str, Any]],
    path: Path,
) -> None:
    """Grava o dashboard.

    Args:
        summary: Resumo de saude.
        batches: Lotes na ordem de execucao.
        path: Destino HTML.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_observability_html(summary, batches), encoding="utf-8")


def _card(label: str, value: str) -> str:
    """Monta um cartao de indicador.

    Args:
        label: Nome do indicador.
        value: Valor ja formatado.

    Returns:
        Fragmento HTML.
    """
    return f"<div class='card'><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"


def _batch_row(batch: dict[str, Any]) -> str:
    """Formata uma linha da tabela de lotes.

    Args:
        batch: Registro agregado do lote.

    Returns:
        Elemento `<tr>`.
    """
    drift = batch.get("drift", {})
    if not drift.get("evaluated"):
        drift_label = "nao avaliado"
    elif drift.get("alert") and not drift.get("confirmed"):
        drift_label = "pagina, nao confirma: " + ", ".join(drift.get("alert_columns", []))
    elif drift.get("alert"):
        drift_label = "confirmado: " + ", ".join(drift.get("confirmed_columns", []))
    else:
        drift_label = "estavel"
    score = batch.get("score_mean")
    return (
        "<tr>"
        f"<td>{escape(str(batch['batch_id']))}</td>"
        f"<td>{escape(str(batch['batch_kind']))}</td>"
        f"<td>{int(batch['accepted_rows'])}</td>"
        f"<td>{int(batch['rejected_rows'])}</td>"
        f"<td>{_percent(float(batch['null_cell_rate']))}</td>"
        f"<td>{_latency(batch.get('latency_ms_per_row'))}</td>"
        f"<td>{_number(score)}</td>"
        f"<td>{escape(drift_label)}</td>"
        "</tr>"
    )


def _drift_table(item: dict[str, Any]) -> str:
    """Mostra so as colunas que sairam da faixa estavel, mais o score.

    Args:
        item: Bloco `drift_monitor` do resumo.

    Returns:
        Tabela HTML.
    """
    interesting = [
        row
        for row in item.get("columns", [])
        if row["column"] == "default_probability" or row["affected"]
    ]
    body = "\n".join(
        "<tr>"
        f"<td>{escape(str(row['column']))}</td>"
        f"<td>{_number(row['psi'])}</td>"
        f"<td>{escape(str(row['psi_band']))}</td>"
        "</tr>"
        for row in interesting
    )
    affected = ", ".join(item.get("affected_columns", [])) or "nenhuma"
    return (
        f"<h3>{escape(str(item.get('label', '')))}</h3>"
        f"<p class='note'>Colunas fora da faixa estavel: {escape(affected)}. "
        f"PSI do score: {_number(item.get('score_psi'))} ({escape(str(item.get('score_psi_band', '')))}).</p>"
        "<table><thead><tr><th>Coluna</th><th>PSI</th><th>Faixa</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _percent(value: float) -> str:
    """Formata proporcao como percentual com virgula.

    Args:
        value: Numero entre 0 e 1, em geral.

    Returns:
        Texto percentual.
    """
    return f"{value * 100:.2f}".replace(".", ",") + "%"


def _latency(value: float | None) -> str:
    """Formata milissegundos.

    Args:
        value: Latencia, ou None se o lote nao teve inferencia.

    Returns:
        Texto curto.
    """
    if value is None:
        return "—"
    return f"{value:.3f}".replace(".", ",")


def _number(value: float | None) -> str:
    """Formata um float agregado.

    Args:
        value: Metrica, ou None.

    Returns:
        Texto com tres casas, ou traco.
    """
    if value is None:
        return "—"
    return f"{float(value):.3f}".replace(".", ",")


def _integer(value: int) -> str:
    """Formata volume com separador de milhar.

    Args:
        value: Contagem.

    Returns:
        Texto com ponto de milhar.
    """
    return f"{int(value):,}".replace(",", ".")
