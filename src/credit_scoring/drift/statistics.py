"""PSI, KS e qui-quadrado usados como evidencia de drift.

Tres ideias que costumam se misturar:

- Data drift: P(X) mudou. O perfil de quem pede credito nao e o mesmo
  da base de treino. PSI e KS medem isso, coluna a coluna.
- Prediction drift: P(score) mudou. Pode acontecer mesmo sem rotulo,
  porque o modelo transforma as features numa probabilidade.
- Concept drift: P(Y|X) mudou. O mesmo perfil passa a inadimplir mais
  ou menos. PSI de feature nao prova isso. A evidencia aqui e a taxa
  de Charged Off dentro da mesma faixa de score.

O KS deste modulo e o teste de duas amostras entre referencia e
producao. Nao e o KS de credito do baseline, que separa adimplentes
de inadimplentes dentro de um unico conjunto.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from credit_scoring.config import (
    KS_ALPHA,
    PSI_BIN_COUNT,
    PSI_MODERATE_THRESHOLD,
    PSI_SIGNIFICANT_THRESHOLD,
)

_EPSILON = 1e-6


def psi_band(psi: float) -> str:
    """Classifica o PSI na escala usada em credito.

    Args:
        psi: Population Stability Index.

    Returns:
        `stable`, `moderate` ou `significant`.
    """
    if psi < PSI_MODERATE_THRESHOLD:
        return "stable"
    if psi < PSI_SIGNIFICANT_THRESHOLD:
        return "moderate"
    return "significant"


def _share(counts: np.ndarray) -> np.ndarray:
    """Converte contagens em proporcoes sem deixar bin vazio.

    Um bin com zero quebra o log do PSI. Somar um epsilon pequeno e
    renormalizar evita divisao por zero sem inventar uma categoria.

    Args:
        counts: Contagem por bin ou categoria.

    Returns:
        Proporcoes que somam 1.
    """
    adjusted = counts.astype(float) + _EPSILON
    return adjusted / adjusted.sum()


def _numeric_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Cria bins pelos quantis da referencia e abre as pontas.

    Os cortes nascem so da referencia. Se nascessem das duas amostras,
    a producao ajudaria a definir a regua com que ela mesma e medida.

    Args:
        reference: Valores numericos da referencia.
        bins: Numero desejado de faixas.

    Returns:
        Bordas prontas para `numpy.histogram`.
    """
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        center = float(edges[0]) if len(edges) else 0.0
        edges = np.array([center - 1.0, center + 1.0])
    edges = edges.astype(float)
    # Inclui valores de producao fora do minimo e do maximo observados.
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = PSI_BIN_COUNT,
) -> float:
    """Calcula o PSI entre duas amostras da mesma variavel.

    PSI = soma (p_producao - p_referencia) * ln(p_producao / p_referencia).
    Zero significa a mesma distribuicao. Nao e um p-valor: a leitura
    depende da faixa 0,10 / 0,25, nao de um teste de hipotese.

    Args:
        reference: Valores da janela de treino.
        current: Valores da janela comparada.
        bins: Faixas de quantil. Ignorado quando a serie e categorica.

    Returns:
        PSI nao negativo.
    """
    reference_values = reference.dropna()
    current_values = current.dropna()
    if reference_values.empty or current_values.empty:
        raise ValueError("PSI exige as duas amostras nao vazias.")

    if pd.api.types.is_numeric_dtype(reference_values) and pd.api.types.is_numeric_dtype(
        current_values
    ):
        edges = _numeric_edges(reference_values.to_numpy(dtype=float), bins)
        reference_counts = np.histogram(reference_values.to_numpy(dtype=float), bins=edges)[0]
        current_counts = np.histogram(current_values.to_numpy(dtype=float), bins=edges)[0]
    else:
        categories = sorted(set(reference_values.astype(str)) | set(current_values.astype(str)))
        reference_counts = (
            reference_values.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()
        )
        current_counts = (
            current_values.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()
        )

    reference_share = _share(np.asarray(reference_counts))
    current_share = _share(np.asarray(current_counts))
    psi = np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    return float(max(psi, 0.0))


def kolmogorov_smirnov(reference: pd.Series, current: pd.Series) -> dict[str, float | bool]:
    """Aplica o KS de duas amostras a uma coluna numerica.

    A hipotese nula e que as duas amostras vem da mesma distribuicao.
    Drift, neste teste, e p-valor abaixo de `KS_ALPHA`. Com amostra
    grande, um deslocamento pequeno demais para importar no credito
    ainda rejeita a nula. Por isso o PSI continua sendo o alerta.

    Args:
        reference: Amostra numerica de referencia.
        current: Amostra numerica atual.

    Returns:
        Estatistica D, p-valor e a decisao no alfa configurado.
    """
    reference_values = reference.dropna().to_numpy(dtype=float)
    current_values = current.dropna().to_numpy(dtype=float)
    if len(reference_values) == 0 or len(current_values) == 0:
        raise ValueError("KS exige as duas amostras nao vazias.")
    statistic, p_value = stats.ks_2samp(reference_values, current_values)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < KS_ALPHA),
    }


def chi_square(reference: pd.Series, current: pd.Series) -> dict[str, float | bool | int]:
    """Compara frequencias categoricas com o teste qui-quadrado.

    E o equivalente categorico do KS: tambem devolve p-valor e tambem
    fica sensivel quando a amostra e grande. A tabela usa as categorias
    das duas janelas, com zero onde uma delas nao aparece.

    Args:
        reference: Categorias da referencia.
        current: Categorias da producao.

    Returns:
        Estatistica, p-valor, graus de liberdade e a decisao no alfa.
    """
    categories = sorted(set(reference.astype(str)) | set(current.astype(str)))
    reference_counts = reference.astype(str).value_counts().reindex(categories, fill_value=0)
    current_counts = current.astype(str).value_counts().reindex(categories, fill_value=0)
    table = np.vstack([reference_counts.to_numpy(), current_counts.to_numpy()])
    if table.shape[1] < 2 or table.sum() == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "degrees_of_freedom": 0,
            "drift_detected": False,
        }
    statistic, p_value, degrees_of_freedom, _ = stats.chi2_contingency(table)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees_of_freedom),
        "drift_detected": bool(p_value < KS_ALPHA),
    }


def _column_result(
    column: str,
    kind: str,
    reference: pd.Series,
    current: pd.Series,
    intentional: bool,
) -> dict[str, Any]:
    """Reune PSI e o teste adequado ao tipo da coluna.

    Args:
        column: Nome da coluna.
        kind: `numeric` ou `categorical`.
        reference: Serie de referencia.
        current: Serie atual.
        intentional: Verdadeiro se a coluna foi alterada pela simulacao.

    Returns:
        Uma linha do relatorio estatistico, sem dados de cliente.
    """
    psi = population_stability_index(reference, current)
    result: dict[str, Any] = {
        "column": column,
        "kind": kind,
        "intentional": intentional,
        "psi": psi,
        "psi_band": psi_band(psi),
        # Alerta operacional: PSI moderado ou maior. O p-valor entra como apoio.
        "affected": psi >= PSI_MODERATE_THRESHOLD,
        "mean_reference": None,
        "mean_current": None,
        "ks_statistic": None,
        "ks_p_value": None,
        "ks_drift_detected": None,
        "chi2_statistic": None,
        "chi2_p_value": None,
        "chi2_drift_detected": None,
    }
    if kind == "numeric":
        result["mean_reference"] = float(reference.mean())
        result["mean_current"] = float(current.mean())
        ks_result = kolmogorov_smirnov(reference, current)
        result["ks_statistic"] = ks_result["statistic"]
        result["ks_p_value"] = ks_result["p_value"]
        result["ks_drift_detected"] = ks_result["drift_detected"]
        return result

    chi_result = chi_square(reference, current)
    result["chi2_statistic"] = chi_result["statistic"]
    result["chi2_p_value"] = chi_result["p_value"]
    result["chi2_drift_detected"] = chi_result["drift_detected"]
    return result


def compare_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_columns: tuple[str, ...] | list[str],
    categorical_columns: tuple[str, ...] | list[str],
    intentional_columns: tuple[str, ...] | list[str] = (),
) -> list[dict[str, Any]]:
    """Compara cada coluna e devolve a evidencia usada no relatorio.

    Args:
        reference: Dataset de referencia, ja sem identificadores.
        current: Dataset atual, com as mesmas colunas comparadas.
        numeric_columns: Colunas que recebem PSI por quantil e KS.
        categorical_columns: Colunas que recebem PSI por categoria e qui-quadrado.
        intentional_columns: Colunas alteradas de proposito na simulacao.

    Returns:
        Lista ordenada pelo PSI, da maior mudanca para a menor.
    """
    intentional = set(intentional_columns)
    rows = [
        _column_result(
            column,
            "numeric",
            reference[column],
            current[column],
            column in intentional,
        )
        for column in numeric_columns
    ]
    rows.extend(
        _column_result(
            column,
            "categorical",
            reference[column],
            current[column],
            column in intentional,
        )
        for column in categorical_columns
    )
    return sorted(rows, key=lambda item: float(item["psi"]), reverse=True)


def calibration_gap(
    reference_scores: pd.Series,
    reference_target: pd.Series,
    current_scores: pd.Series,
    current_target: pd.Series,
    bands: int = 5,
) -> dict[str, Any]:
    """Mede se a mesma faixa de score continua com o mesmo default.

    As faixas sao quantis do score da referencia. Em cada faixa, compara
    a taxa observada de Charged Off. Uma diferenca grande, com features
    ainda parecidas, e o sinal de concept drift. Se as features tambem
    mudaram, a diferenca pode ser so migracao de score.

    Args:
        reference_scores: Probabilidade prevista na referencia.
        reference_target: Default observado na referencia.
        current_scores: Probabilidade prevista na producao.
        current_target: Default observado na producao.
        bands: Numero de faixas de score.

    Returns:
        Taxas por faixa e a distancia absoluta media, ponderada pelo
        volume da producao.
    """
    edges = np.unique(np.quantile(reference_scores.to_numpy(dtype=float), np.linspace(0, 1, bands + 1)))
    if len(edges) < 2:
        return {"bands": [], "weighted_absolute_gap": 0.0}
    edges = edges.astype(float)
    edges[0] = 0.0
    edges[-1] = 1.0
    reference_band = pd.cut(reference_scores, bins=edges, include_lowest=True)
    current_band = pd.cut(current_scores, bins=edges, include_lowest=True)
    details: list[dict[str, Any]] = []
    weighted_gap = 0.0
    current_total = 0
    for band in reference_band.cat.categories:
        reference_mask = reference_band.eq(band)
        current_mask = current_band.eq(band)
        current_count = int(current_mask.sum())
        reference_rate = (
            float(reference_target.loc[reference_mask].mean()) if reference_mask.any() else None
        )
        current_rate = float(current_target.loc[current_mask].mean()) if current_mask.any() else None
        gap = None
        if reference_rate is not None and current_rate is not None:
            gap = abs(current_rate - reference_rate)
            weighted_gap += gap * current_count
            current_total += current_count
        details.append(
            {
                "band": str(band),
                "reference_rate": reference_rate,
                "current_rate": current_rate,
                "absolute_gap": gap,
                "current_count": current_count,
            }
        )
    return {
        "bands": details,
        "weighted_absolute_gap": float(weighted_gap / current_total) if current_total else 0.0,
    }
