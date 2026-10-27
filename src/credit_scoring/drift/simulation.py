"""Monta a producao e aplica o drift intencional dentro do contrato.

A etapa 1 separa dado invalido de dado aceito. A etapa 2 parte do dado
aceito e muda a distribuicao. Um FICO 40 pontos menor continua sendo
um FICO valido; o contrato nao deve barra-lo. Quem precisa perceber a
mudanca e o monitor de drift, nao o Pandera.

Ha dois conjuntos de producao:

- natural: originacoes de 2016 a 2018, sem alteracao artificial. Mostra
  a passagem do tempo.
- drifted: a mesma amostra, com FICO e DTI deslocados. Mostra as duas
  variaveis alteradas de proposito.

O modelo nunca recebe `issue_d` nem `loan_status`. O desfecho fica na
tabela apenas para o diagnostico de concept drift, depois que o score
ja foi calculado.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from credit_scoring.config import (
    DRIFT_DTI_SCALE,
    DRIFT_DTI_SHIFT,
    DRIFT_FICO_SHIFT,
    FEATURE_COLUMNS,
    INTENTIONAL_DRIFT_COLUMNS,
    MIN_PRODUCTION_ROWS,
    PRODUCTION_DRIFTED_PATH,
    PRODUCTION_END,
    PRODUCTION_NATURAL_PATH,
    PRODUCTION_SAMPLE_SIZE,
    PRODUCTION_START,
    RAW_LOANS_PATH,
)
from credit_scoring.data.prepare import (
    extract_window_candidates,
    normalize_application_fields,
    stratified_sample,
)
from credit_scoring.data.schema import PRODUCTION_SCHEMA, quarantine_invalid_rows
from credit_scoring.models.baseline import defaulted_target, predict_default_probability


def assign_production_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Coloca datas sinteticas dentro da janela de producao.

    Serve aos testes e ao modo sem arquivo bruto. Nao substitui a
    originacao real de 2016-2018 quando o CSV do Lending Club existe.

    Args:
        frame: Amostra que ja tem as colunas do contrato.

    Returns:
        Copia com `issue_d` espalhado entre 2016 e 2018.
    """
    dated = frame.copy()
    start = pd.Timestamp(PRODUCTION_START)
    dated["issue_d"] = [start + pd.DateOffset(months=index % 36) for index in range(len(dated))]
    return dated


def apply_intentional_drift(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Desloca FICO e DTI e recorta os valores para o contrato continuar valido.

    O clip e parte da simulacao, nao um conserto silencioso. Sem ele,
    um DTI deslocado poderia passar de 100 e a etapa 2 viraria uma nova
    falha de contrato. A pergunta desta etapa e outra: o dado valido
    ainda parece com o dado de treino?

    Args:
        frame: Producao natural, ja aceita pelo contrato.

    Returns:
        Amostra deslocada e o registro agregado da transformacao.
    """
    drifted = frame.copy()
    fico_before = drifted["fico_range_low"].astype(float)
    fico_raw = fico_before + DRIFT_FICO_SHIFT
    fico_after = fico_raw.clip(lower=300.0, upper=850.0)
    drifted["fico_range_low"] = fico_after

    dti_before = drifted["dti"].astype(float)
    dti_raw = dti_before * DRIFT_DTI_SCALE + DRIFT_DTI_SHIFT
    dti_after = dti_raw.clip(lower=0.0, upper=100.0)
    drifted["dti"] = dti_after

    report = {
        "intentional_columns": list(INTENTIONAL_DRIFT_COLUMNS),
        "transformations": [
            {
                "column": "fico_range_low",
                "operation": "additive_shift",
                "delta": DRIFT_FICO_SHIFT,
                "clip": [300.0, 850.0],
                "business_meaning": "Queda do score FICO, perfil de credito pior.",
                "mean_before": float(fico_before.mean()),
                "mean_after": float(fico_after.mean()),
                "rows_clipped": int((fico_after != fico_raw).sum()),
            },
            {
                "column": "dti",
                "operation": "scale_and_shift",
                "scale": DRIFT_DTI_SCALE,
                "delta": DRIFT_DTI_SHIFT,
                "clip": [0.0, 100.0],
                "business_meaning": "Aumento do comprometimento de renda.",
                "mean_before": float(dti_before.mean()),
                "mean_after": float(dti_after.mean()),
                "rows_clipped": int((dti_after != dti_raw).sum()),
            },
        ],
        "privacy": {
            "row_level_values_persisted": False,
            "identifiers_excluded": True,
        },
    }
    return drifted, report


def _prepare_production_frame(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normaliza, aplica o contrato de producao e reduz a amostra.

    Args:
        candidates: Linhas da janela, ainda nao contratadas.

    Returns:
        Amostra natural e o resumo da quarentena.

    Raises:
        RuntimeError: Se a amostra ficar abaixo do minimo do desafio.
    """
    normalized = normalize_application_fields(candidates)
    accepted, quarantine = quarantine_invalid_rows(normalized, PRODUCTION_SCHEMA)
    sampled = stratified_sample(accepted, sample_size=PRODUCTION_SAMPLE_SIZE)
    if len(sampled) < MIN_PRODUCTION_ROWS:
        raise RuntimeError(
            "Amostra de producao abaixo do minimo de "
            f"{MIN_PRODUCTION_ROWS} linhas apos o contrato: {len(sampled)}."
        )
    return sampled, quarantine


def prepare_production_from_raw(
    raw_path: Path = RAW_LOANS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extrai a producao natural da janela posterior a referencia.

    Args:
        raw_path: CSV compactado do Lending Club.

    Returns:
        Amostra natural e contagens da leitura, sem dados de linha.
    """
    candidates, counts = extract_window_candidates(
        raw_path=raw_path,
        window_start=PRODUCTION_START,
        window_end=PRODUCTION_END,
    )
    sampled, quarantine = _prepare_production_frame(candidates)
    report = {
        "source": "raw_window",
        "window": {"start": PRODUCTION_START, "end": PRODUCTION_END},
        "rows_read": counts["rows_read"],
        "rows_eligible": counts["rows_eligible"],
        "rows_after_fractional_sample": counts["rows_after_fractional_sample"],
        "rows_production_sample": int(len(sampled)),
        "quarantine": quarantine,
    }
    return sampled, report


def prepare_production_from_reference(reference: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reaproveita a referencia so quando o bruto nao puder ser lido.

    Muda a data para a janela de producao e embaralha a amostra. Nao
    injeta drift: isso fica em `apply_intentional_drift`, para as duas
    fontes terem o mesmo segundo passo.

    Args:
        reference: Base de referencia ja validada.

    Returns:
        Producao natural sintetica e um relatorio que deixa a origem explicita.
    """
    renamed = assign_production_dates(reference)
    sampled, quarantine = _prepare_production_frame(renamed)
    report = {
        "source": "reference_replay",
        "window": {"start": PRODUCTION_START, "end": PRODUCTION_END},
        "rows_production_sample": int(len(sampled)),
        "quarantine": quarantine,
        "limitation": (
            "Sem o arquivo bruto, a passagem do tempo e apenas a troca da data. "
            "O drift natural de 2016-2018 nao aparece nesta origem."
        ),
    }
    return sampled, report


def save_production_dataset(frame: pd.DataFrame, path: Path) -> None:
    """Grava a producao em `data/`, que permanece fora do Git.

    Args:
        frame: Amostra natural ou deslocada.
        path: Destino CSV.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exportable = frame.copy()
    exportable["issue_d"] = pd.to_datetime(exportable["issue_d"]).dt.strftime("%Y-%m-%d")
    exportable.to_csv(destination, index=False)


def load_production_dataset(path: Path) -> pd.DataFrame:
    """Carrega uma amostra de producao e exige o contrato antes de devolve-la.

    Args:
        path: CSV gerado por `save_production_dataset`.

    Returns:
        Base validada pelo contrato de producao.

    Raises:
        FileNotFoundError: Se a simulacao ainda nao foi executada.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Amostra de producao ausente em {source}. Execute a simulacao da etapa 2."
        )
    frame = pd.read_csv(source)
    frame["issue_d"] = pd.to_datetime(frame["issue_d"])
    return PRODUCTION_SCHEMA.validate(frame, lazy=True)


def score_applications(artifact: dict[str, Any], frame: pd.DataFrame) -> pd.Series:
    """Gera a probabilidade de default sem entregar o desfecho ao modelo.

    Args:
        artifact: Baseline carregado ou treinado.
        frame: Referencia ou producao, com ou sem colunas de metadado.

    Returns:
        Probabilidade de Charged Off, alinhada ao indice do frame.
    """
    features = frame.loc[:, list(FEATURE_COLUMNS)]
    return predict_default_probability(artifact, features)


def prediction_summary(scores: pd.Series) -> dict[str, float | int]:
    """Resume um lote de scores sem persistir probabilidade por cliente.

    Args:
        scores: Saida de `score_applications`.

    Returns:
        Volume e quantis do lote.
    """
    return {
        "count": int(scores.shape[0]),
        "mean": float(scores.mean()),
        "p50": float(scores.quantile(0.50)),
        "p95": float(scores.quantile(0.95)),
    }


def observed_default_rate(frame: pd.DataFrame) -> float:
    """Calcula a taxa de Charged Off de uma janela com desfecho.

    Args:
        frame: Referencia ou producao com `loan_status`.

    Returns:
        Proporcao da classe positiva.
    """
    return float(defaulted_target(frame["loan_status"]).mean())


def production_paths() -> dict[str, Path]:
    """Devolve os destinos padrao da simulacao.

    Returns:
        Caminhos da producao natural e da producao deslocada.
    """
    return {
        "natural": PRODUCTION_NATURAL_PATH,
        "drifted": PRODUCTION_DRIFTED_PATH,
    }
