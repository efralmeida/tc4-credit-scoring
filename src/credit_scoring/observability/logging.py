"""Logs de predicao sem feature, identificador ou payload rejeitado.

O log de lote e agregado e pode ir para o repositorio. O log de evento
guarda faixa de score e um id aleatorio, que nao e chave de cliente.
Ele fica fora do Git. Nenhum dos dois grava a linha recusada.
"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from credit_scoring.config import EXCLUDED_IDENTIFIERS, FEATURE_COLUMNS

FORBIDDEN_LOG_FIELDS = set(FEATURE_COLUMNS) | set(EXCLUDED_IDENTIFIERS) | {
    "loan_status",
    "issue_d",
    "default_probability",
    "member_id",
}


def utc_now() -> str:
    """Devolve o instante atual em ISO-8601, com fuso UTC.

    Returns:
        Timestamp usado nos logs operacionais.
    """
    return datetime.now(UTC).isoformat()


def new_event_id() -> str:
    """Gera um id de evento que nao pode ser juntado ao arquivo bruto.

    Returns:
        UUID4 em texto.
    """
    return str(uuid.uuid4())


def score_bucket(score: float) -> str:
    """Agrupa a probabilidade em faixas de 0,2.

    A faixa entra no log. A probabilidade exata nao entra. Um score
    unico facilitaria reidentificar a solicitacao se alguem cruzasse
    o log com outra base.

    Args:
        score: Probabilidade estimada de Charged Off.

    Returns:
        Rotulo da faixa, no formato `0.4-0.6`.
    """
    clipped = min(max(float(score), 0.0), 0.999999)
    lower = int(clipped * 5) / 5
    return f"{lower:.1f}-{lower + 0.2:.1f}"


def assert_log_minimized(record: dict[str, Any]) -> None:
    """Recusa um log que carregue feature ou identificador.

    Args:
        record: Evento ou lote prestes a ser gravado.

    Raises:
        ValueError: Se alguma chave proibida estiver presente.
    """
    leaked = FORBIDDEN_LOG_FIELDS & set(record)
    if leaked:
        joined = ", ".join(sorted(leaked))
        raise ValueError(f"Log operacional nao pode carregar: {joined}.")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Grava registros JSONL, um por linha, depois de checar a minimizacao.

    Args:
        path: Destino do log.
        records: Eventos ou lotes ja sem dado de cliente.
    """
    for record in records:
        assert_log_minimized(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    text = "\n".join(lines)
    if text:
        text += "\n"
    destination.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Le um log JSONL.

    Args:
        path: Arquivo gerado por `write_jsonl`.

    Returns:
        Lista de registros. Arquivo ausente devolve lista vazia.
    """
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
