"""Relatorio Evidently e resumo estatistico da etapa 2.

O HTML e a evidencia visual pedida pelo desafio. O JSON ao lado explica
a evidencia numerica, porque o grafico sozinho nao diz qual teste foi
usado nem por que o PSI manda no alerta.

O Evidently, com mais de mil linhas de referencia, trocaria o KS pelo
Wasserstein. Aqui o metodo e fixado em PSI para o relatorio ficar
alinhado ao teste que o credito ja usa. O KS de duas amostras continua
no JSON, calculado por nos, justamente para nao depender dessa troca
automatica.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset

from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    DRIFT_INJECTION_REPORT_PATH,
    DRIFT_REPORT_PATH,
    EVIDENTLY_SAMPLE_SIZE,
    FEATURE_COLUMNS,
    INTENTIONAL_DRIFT_COLUMNS,
    NUMERIC_FEATURES,
    ROOT,
    PSI_MODERATE_THRESHOLD,
    PSI_SIGNIFICANT_THRESHOLD,
    RANDOM_STATE,
)
from credit_scoring.drift.statistics import calibration_gap, compare_columns, population_stability_index
from credit_scoring.models.baseline import defaulted_target

PREDICTION_COLUMN = "default_probability"
TARGET_COLUMN = "defaulted"


def monitoring_frame(frame: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """Monta a tabela que pode entrar no monitor, sem identificador.

    `issue_d` fica de fora de proposito. Se entrasse, o relatorio
    apontaria drift de data so porque as janelas sao diferentes, e isso
    esconderia a pergunta sobre as features.

    Args:
        frame: Referencia ou producao validada.
        scores: Probabilidades do baseline para as mesmas linhas.

    Returns:
        Features, score e desfecho binario.
    """
    monitored = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    monitored[PREDICTION_COLUMN] = scores.to_numpy()
    monitored[TARGET_COLUMN] = defaulted_target(frame["loan_status"]).to_numpy()
    return monitored


def sample_pair(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    sample_size: int = EVIDENTLY_SAMPLE_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduz os dois lados do HTML sem mudar o teste estatistico completo.

    Args:
        reference: Tabela de referencia ja no formato de monitoramento.
        current: Tabela atual no mesmo formato.
        sample_size: Maximo de linhas enviado ao Evidently.

    Returns:
        Par amostrado, com a mesma semente dos outros artefatos.
    """
    reference_sample = reference
    current_sample = current
    if len(reference_sample) > sample_size:
        reference_sample = reference_sample.sample(n=sample_size, random_state=RANDOM_STATE)
    if len(current_sample) > sample_size:
        current_sample = current_sample.sample(n=sample_size, random_state=RANDOM_STATE)
    return reference_sample.reset_index(drop=True), current_sample.reset_index(drop=True)


def _dataset(frame: pd.DataFrame) -> Dataset:
    """Mapeia tipos para o Evidently nao tratar FICO como categoria.

    Args:
        frame: Tabela de monitoramento.

    Returns:
        Dataset com features, predicao e alvo declarados.
    """
    definition = DataDefinition(
        numerical_columns=[*NUMERIC_FEATURES, PREDICTION_COLUMN],
        categorical_columns=[*CATEGORICAL_FEATURES, TARGET_COLUMN],
    )
    return Dataset.from_pandas(frame, data_definition=definition)


def _drift_report() -> Report:
    """Cria o relatorio visual com PSI explicito e KS nas colunas injetadas.

    Returns:
        Report ainda nao executado.
    """
    return Report(
        metrics=[
            DataDriftPreset(
                columns=[*FEATURE_COLUMNS, PREDICTION_COLUMN, TARGET_COLUMN],
                method="psi",
                num_method="psi",
                cat_method="psi",
                # Duas colunas em dezenove nao passam de 50%. O alerta do
                # desafio e por variavel, entao o corte do dataset fica baixo.
                drift_share=0.1,
            ),
            ValueDrift(column="fico_range_low", method="ks"),
            ValueDrift(column="dti", method="ks"),
            ValueDrift(column=PREDICTION_COLUMN, method="ks"),
        ],
        include_tests=True,
    )


def render_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    """Gera o HTML e devolve so o ponteiro do artefato.

    O dicionario completo do Evidently pode carregar amostras. Ele nao
    e gravado. A evidencia numerica reaproveitada no Git e a que nos
    calculamos em `statistics.py`.

    Args:
        reference: Tabela de monitoramento da referencia.
        current: Tabela de monitoramento atual.
        path: Destino HTML.

    Returns:
        Caminho e tamanho do HTML gerado.
    """
    reference_sample, current_sample = sample_pair(reference, current)
    snapshot = _drift_report().run(
        current_data=_dataset(current_sample),
        reference_data=_dataset(reference_sample),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(destination))
    try:
        stored_path = destination.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        stored_path = str(destination)
    return {
        "path": stored_path,
        "bytes": int(destination.stat().st_size),
        "rows_reference_in_html": int(len(reference_sample)),
        "rows_current_in_html": int(len(current_sample)),
    }


def build_summary(
    reference: pd.DataFrame,
    natural: pd.DataFrame,
    drifted: pd.DataFrame,
    reference_scores: pd.Series,
    natural_scores: pd.Series,
    drifted_scores: pd.Series,
    simulation_report: dict[str, Any],
    injection_report: dict[str, Any],
) -> dict[str, Any]:
    """Junta passagem do tempo, injecao, scores e concept drift.

    A comparacao que responde "quais variaveis foram alteradas de
    proposito" e natural contra drifted. A comparacao que responde
    "o modelo ainda ve o mundo do treino" e referencia contra drifted.

    Args:
        reference: Referencia validada.
        natural: Producao antes da injecao.
        drifted: Producao depois da injecao.
        reference_scores: Scores da referencia.
        natural_scores: Scores da producao natural.
        drifted_scores: Scores da producao deslocada.
        simulation_report: Origem e contagens da producao natural.
        injection_report: Transformacoes de FICO e DTI.

    Returns:
        Resumo serializavel, sem linha de cliente.
    """
    reference_monitor = monitoring_frame(reference, reference_scores)
    natural_monitor = monitoring_frame(natural, natural_scores)
    drifted_monitor = monitoring_frame(drifted, drifted_scores)
    time_columns = compare_columns(
        reference_monitor,
        natural_monitor,
        list(NUMERIC_FEATURES),
        list(CATEGORICAL_FEATURES),
    )
    injection_columns = compare_columns(
        natural_monitor,
        drifted_monitor,
        list(NUMERIC_FEATURES),
        list(CATEGORICAL_FEATURES),
        intentional_columns=INTENTIONAL_DRIFT_COLUMNS,
    )
    monitoring_columns = compare_columns(
        reference_monitor,
        drifted_monitor,
        [*NUMERIC_FEATURES, PREDICTION_COLUMN],
        [*CATEGORICAL_FEATURES, TARGET_COLUMN],
        intentional_columns=INTENTIONAL_DRIFT_COLUMNS,
    )
    concept = calibration_gap(
        reference_scores,
        reference_monitor[TARGET_COLUMN],
        natural_scores,
        natural_monitor[TARGET_COLUMN],
    )
    return {
        "evidence": {
            "psi": (
                "PSI compara proporcoes por faixa. Abaixo de "
                f"{PSI_MODERATE_THRESHOLD} a coluna fica estavel; a partir de "
                f"{PSI_SIGNIFICANT_THRESHOLD} o deslocamento e relevante para credito."
            ),
            "ks": (
                "KS de duas amostras testa se referencia e producao podem vir "
                "da mesma distribuicao continua. O p-valor menor que 0,05 rejeita "
                "essa hipotese, mas amostra grande rejeita diferenca pequena."
            ),
            "chi_square": (
                "Qui-quadrado e o teste equivalente para colunas categoricas. "
                "Tambem devolve p-valor e tem a mesma ressalva de amostra grande."
            ),
            "concept_drift": (
                "Concept drift e lido na producao natural, nao na amostra injetada. "
                "Compara a taxa de Charged Off dentro da mesma faixa de score."
            ),
        },
        "simulation": simulation_report,
        "injection": injection_report,
        "affected_by_injection": [
            row["column"] for row in injection_columns if row["affected"] and row["intentional"]
        ],
        "unexpected_injection_drift": [
            row["column"] for row in injection_columns if row["affected"] and not row["intentional"]
        ],
        "affected_against_reference": [row["column"] for row in monitoring_columns if row["affected"]],
        "time_passage_columns": time_columns,
        "injection_columns": injection_columns,
        "reference_vs_drifted_columns": monitoring_columns,
        "prediction_drift": {
            "psi_natural_vs_reference": population_stability_index(reference_scores, natural_scores),
            "psi_drifted_vs_reference": population_stability_index(reference_scores, drifted_scores),
            "psi_drifted_vs_natural": population_stability_index(natural_scores, drifted_scores),
        },
        "concept_drift_natural_vs_reference": concept,
        "privacy": {
            "identifiers_in_report": False,
            "row_level_scores_persisted": False,
            "evidently_dict_persisted": False,
        },
    }


def write_stage2_reports(
    reference: pd.DataFrame,
    natural: pd.DataFrame,
    drifted: pd.DataFrame,
    reference_scores: pd.Series,
    natural_scores: pd.Series,
    drifted_scores: pd.Series,
    simulation_report: dict[str, Any],
    injection_report: dict[str, Any],
    monitoring_html: Path = DRIFT_REPORT_PATH,
    injection_html: Path = DRIFT_INJECTION_REPORT_PATH,
) -> dict[str, Any]:
    """Grava os dois HTML e devolve o JSON que explica os testes.

    Args:
        reference: Referencia validada.
        natural: Producao antes da injecao.
        drifted: Producao depois da injecao.
        reference_scores: Scores da referencia.
        natural_scores: Scores da producao natural.
        drifted_scores: Scores da producao deslocada.
        simulation_report: Origem da producao.
        injection_report: Transformacoes intencionais.
        monitoring_html: HTML de referencia contra producao deslocada.
        injection_html: HTML que isola FICO e DTI.

    Returns:
        Resumo completo, incluindo os ponteiros dos HTML.
    """
    summary = build_summary(
        reference,
        natural,
        drifted,
        reference_scores,
        natural_scores,
        drifted_scores,
        simulation_report,
        injection_report,
    )
    summary["reports"] = {
        "reference_vs_drifted": render_drift_report(
            monitoring_frame(reference, reference_scores),
            monitoring_frame(drifted, drifted_scores),
            monitoring_html,
        ),
        "natural_vs_injected": render_drift_report(
            monitoring_frame(natural, natural_scores),
            monitoring_frame(drifted, drifted_scores),
            injection_html,
        ),
    }
    return summary
