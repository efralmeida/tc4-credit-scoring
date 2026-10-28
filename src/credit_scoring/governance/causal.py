"""Separa causa desenhada de associacao observada.

Ha dois contrastes e eles nao dizem a mesma coisa.

A injecao de FICO e DTI e uma intervencao nas mesmas linhas. O rotulo
nao muda. Se o score se move, a causa do prediction drift e essa
intervencao. Isso nao e concept drift.

A janela 2016-2018 e observacional. A decomposicao de Kitagawa separa
quanto da taxa de Charged Off veio da mudanca de mistura do score e
quanto veio da taxa dentro da mesma faixa. Isso nao identifica a causa
macroeconomica. Identifica onde o efeito aparece.
"""

from typing import Any

import numpy as np
import pandas as pd

from credit_scoring.drift.statistics import population_stability_index
from credit_scoring.models.baseline import defaulted_target


def kitagawa_decomposition(
    reference_scores: pd.Series,
    reference_target: pd.Series,
    current_scores: pd.Series,
    current_target: pd.Series,
    bands: int = 5,
) -> dict[str, Any]:
    """Decompoe a variacao da taxa de default em mistura e taxa.

    As faixas nascem dos quantis do score de referencia, como no gap de
    calibracao da etapa 2. Composicao usa a taxa da referencia. Efeito
    de taxa usa o peso da producao.

    Args:
        reference_scores: Score da referencia.
        reference_target: Default da referencia.
        current_scores: Score atual.
        current_target: Default atual.
        bands: Numero de faixas.

    Returns:
        Efeitos, residuo e a leitura permitida.
    """
    edges = np.unique(np.quantile(reference_scores.to_numpy(dtype=float), np.linspace(0, 1, bands + 1)))
    edges = edges.astype(float)
    if len(edges) < 2:
        return _empty_decomposition()
    edges[0] = 0.0
    edges[-1] = 1.0
    reference_band = pd.cut(reference_scores, bins=edges, include_lowest=True)
    current_band = pd.cut(current_scores, bins=edges, include_lowest=True)
    composition = 0.0
    rate_effect = 0.0
    details: list[dict[str, Any]] = []
    n_reference = len(reference_scores)
    n_current = len(current_scores)
    for band in reference_band.cat.categories:
        reference_mask = reference_band.eq(band)
        current_mask = current_band.eq(band)
        weight_reference = float(reference_mask.sum() / n_reference)
        weight_current = float(current_mask.sum() / n_current)
        rate_reference = (
            float(reference_target.loc[reference_mask].mean()) if reference_mask.any() else None
        )
        rate_current = float(current_target.loc[current_mask].mean()) if current_mask.any() else None
        composition_part = (weight_current - weight_reference) * (rate_reference or 0.0)
        rate_part = 0.0
        if rate_reference is not None and rate_current is not None:
            rate_part = weight_current * (rate_current - rate_reference)
        composition += composition_part
        rate_effect += rate_part
        details.append(
            {
                "band": str(band),
                "weight_reference": weight_reference,
                "weight_current": weight_current,
                "rate_reference": rate_reference,
                "rate_current": rate_current,
                "composition_effect": composition_part,
                "rate_effect": rate_part,
            }
        )
    observed = float(current_target.mean() - reference_target.mean())
    return {
        "method": "kitagawa",
        "bands": details,
        "observed_delta": observed,
        "composition_effect": float(composition),
        "rate_effect": float(rate_effect),
        "residual": float(observed - composition - rate_effect),
        "reading": (
            "Efeito de taxa e mudanca de P(Y|faixa de score): o sinal de concept drift. "
            "Efeito de composicao e migracao entre faixas. Nenhum dos dois nomeia a causa externa."
        ),
    }


def mix_standardization(
    reference_group: pd.Series,
    reference_target: pd.Series,
    current_group: pd.Series,
    current_target: pd.Series,
) -> dict[str, Any]:
    """Ve se a mudanca de mix de uma categoria explica a taxa de default.

    A taxa padronizada aplica o mix da referencia sobre as taxas da
    producao. Se o gap continuar, aquela categoria nao da conta da
    diferenca.

    Args:
        reference_group: Categoria na referencia.
        reference_target: Default na referencia.
        current_group: Categoria na producao.
        current_target: Default na producao.

    Returns:
        Taxa observada, taxa padronizada e o gap que sobra.
    """
    reference = pd.DataFrame(
        {"group": reference_group.astype(str), "target": reference_target.to_numpy()}
    )
    current = pd.DataFrame({"group": current_group.astype(str), "target": current_target.to_numpy()})
    reference_rate = float(reference["target"].mean())
    current_rate = float(current["target"].mean())
    weights = reference["group"].value_counts(normalize=True)
    current_rates = current.groupby("group")["target"].mean()
    common = [group for group in weights.index if group in current_rates.index]
    if not common:
        standardized = None
        remaining = None
    else:
        used = weights.loc[common]
        used = used / used.sum()
        standardized = float((used * current_rates.loc[common]).sum())
        remaining = float(standardized - reference_rate)
    return {
        "reference_rate": reference_rate,
        "current_rate": current_rate,
        "current_rate_at_reference_mix": standardized,
        "gap_if_mix_unchanged": remaining,
        "reading": (
            "Se o gap padronizado continua perto do gap observado, a troca de mix "
            "dessa categoria nao explica a alta do default."
        ),
    }


def intervention_contrast(
    natural_scores: pd.Series,
    drifted_scores: pd.Series,
    natural_target: pd.Series,
    drifted_target: pd.Series,
) -> dict[str, Any]:
    """Atribui o movimento do score a intervencao conhecida.

    Args:
        natural_scores: Score antes de mexer em FICO e DTI.
        drifted_scores: Score das mesmas linhas, depois da intervencao.
        natural_target: Default antes.
        drifted_target: Default depois. A simulacao nao altera o rotulo.

    Returns:
        Contraste causal do desenho, sem extrapolar para a economia.
    """
    score_psi = population_stability_index(natural_scores, drifted_scores)
    return {
        "design": "mesmas linhas, intervencao conhecida em fico_range_low e dti",
        "intervened_columns": ["fico_range_low", "dti"],
        "score_mean_before": float(natural_scores.mean()),
        "score_mean_after": float(drifted_scores.mean()),
        "score_mean_delta": float(drifted_scores.mean() - natural_scores.mean()),
        "score_psi": score_psi,
        "label_rate_delta": float(drifted_target.mean() - natural_target.mean()),
        "claim": (
            "O prediction drift deste contraste e causado pela queda de FICO e pela alta de DTI. "
            "O rotulo nao mudou, entao o contraste nao e concept drift."
        ),
        "not_claimed": "A intervencao nao explica a janela real de 2016-2018.",
    }


def build_causal_analysis(
    reference: pd.DataFrame,
    natural: pd.DataFrame,
    drifted: pd.DataFrame,
    reference_scores: pd.Series,
    natural_scores: pd.Series,
    drifted_scores: pd.Series,
) -> dict[str, Any]:
    """Junta o contraste experimental e a decomposicao observacional.

    Args:
        reference: Referencia validada.
        natural: Producao sem injecao.
        drifted: Producao com FICO e DTI deslocados.
        reference_scores: Scores da referencia.
        natural_scores: Scores da producao natural.
        drifted_scores: Scores da producao deslocada.

    Returns:
        Analise serializavel, com hipotese separada de achado.
    """
    reference_target = defaulted_target(reference["loan_status"])
    natural_target = defaulted_target(natural["loan_status"])
    drifted_target = defaulted_target(drifted["loan_status"])
    time_gap = kitagawa_decomposition(
        reference_scores,
        reference_target,
        natural_scores,
        natural_target,
    )
    application_mix = mix_standardization(
        reference["application_type"],
        reference_target,
        natural["application_type"],
        natural_target,
    )
    return {
        "question": (
            "O que causou o alerta: dado invalido, intervencao nas features, "
            "ou mudanca da relacao entre score e default?"
        ),
        "invalid_data": (
            "Nao. O contrato ja tirou o impossivel. FICO menor e DTI maior, "
            "dentro da faixa, sao dados validos."
        ),
        "intervention": intervention_contrast(
            natural_scores,
            drifted_scores,
            natural_target,
            drifted_target,
        ),
        "observational_window": {
            "contrast": "referencia 2012-2015 contra producao natural 2016-2018",
            "default_rate_reference": float(reference_target.mean()),
            "default_rate_natural": float(natural_target.mean()),
            "kitagawa": time_gap,
            "application_type_mix": application_mix,
            "claim": (
                "Da para localizar o aumento do default na taxa dentro da faixa de score. "
                "Nao da para apontar um unico fator externo como causa."
            ),
        },
        "hypotheses_not_identified": [
            "Cenario de credito mais duro entre 2016 e 2018.",
            "Mudanca de politica de concessao do Lending Club, nao observada nestas colunas.",
            "Mix de application_type, testado a parte e nao tratado como causa enquanto o gap padronizado continuar.",
            "revol_util no limite do PSI moderado. Associacao, nao mecanismo identificado.",
        ],
    }


def _empty_decomposition() -> dict[str, Any]:
    """Devolve a decomposicao vazia quando o score nao tem variacao.

    Returns:
        Estrutura compativel, com efeitos zerados.
    """
    return {
        "method": "kitagawa",
        "bands": [],
        "observed_delta": 0.0,
        "composition_effect": 0.0,
        "rate_effect": 0.0,
        "residual": 0.0,
        "reading": "Score sem variacao suficiente para abrir faixas.",
    }
