"""Monitor de recorte. Nao e um teste legal de discriminacao.

O modelo nao recebe sexo, raca, idade, religiao, biometria, CEP ou UF.
O que da para ver sao proxies que ficaram de proposito: posse do imovel,
finalidade, tipo de aplicacao, verificacao de renda e tempo de emprego.

O corte de 0,50 nao aprova credito. Ele so separa massa de score baixo
e alto para comparar grupos. A regra dos quatro quintos entra como sinal
de monitoramento, nao como conclusao juridica.
"""

from typing import Any

import pandas as pd

from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    FAIRNESS_DISPARATE_IMPACT,
    FAIRNESS_MIN_GROUP_SIZE,
    FAIRNESS_SCORE_CUTOFF,
)


def audit_slices(
    groups: pd.DataFrame,
    scores: pd.Series,
    target: pd.Series | None = None,
    columns: tuple[str, ...] | list[str] = CATEGORICAL_FEATURES,
    cutoff: float = FAIRNESS_SCORE_CUTOFF,
    min_group_size: int = FAIRNESS_MIN_GROUP_SIZE,
) -> dict[str, Any]:
    """Compara massa de score e default entre categorias.

    Args:
        groups: Tabela com as colunas categoricas auditadas.
        scores: Probabilidade do baseline, no mesmo indice.
        target: Default observado. Quando ausente, a taxa fica vazia.
        columns: Recortes a calcular.
        cutoff: Score abaixo do corte conta como desfecho favoravel ao titular.
        min_group_size: Abaixo disso o grupo aparece, mas nao gera sinal.

    Returns:
        Recortes, sinais e o aviso de que o corte nao e politica de credito.
    """
    aligned_scores = scores.reset_index(drop=True)
    aligned_groups = groups.reset_index(drop=True)
    aligned_target = None if target is None else target.reset_index(drop=True)
    slices = [
        _slice(aligned_groups[column], aligned_scores, aligned_target, column, cutoff, min_group_size)
        for column in columns
        if column in aligned_groups.columns
    ]
    signals = [signal for item in slices for signal in item["signals"]]
    return {
        "cutoff": cutoff,
        "cutoff_meaning": (
            "Corte de monitoramento. Score abaixo dele e tratado como desfecho "
            "favoravel so para comparar grupos. Nao e aprovacao."
        ),
        "disparate_impact_threshold": FAIRNESS_DISPARATE_IMPACT,
        "min_group_size": min_group_size,
        "protected_attributes_in_model": False,
        "excluded_geographic_proxy": ["zip_code", "addr_state"],
        "slices": slices,
        "signals": signals,
        "mitigation": _mitigation(),
    }


def _slice(
    labels: pd.Series,
    scores: pd.Series,
    target: pd.Series | None,
    column: str,
    cutoff: float,
    min_group_size: int,
) -> dict[str, Any]:
    """Calcula um recorte e compara com o grupo de maior volume.

    Args:
        labels: Categorias.
        scores: Scores alinhados.
        target: Default alinhado, ou None.
        column: Nome do recorte.
        cutoff: Corte de monitoramento.
        min_group_size: Volume minimo para sinal.

    Returns:
        Grupos e sinais daquele recorte.
    """
    frame = pd.DataFrame({"group": labels.astype(str), "score": scores.to_numpy()})
    if target is not None:
        frame["target"] = target.to_numpy()
    reference_name = str(frame["group"].value_counts().idxmax())
    reference_rate = float(frame.loc[frame["group"].eq(reference_name), "score"].lt(cutoff).mean())
    rows: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    for name, part in frame.groupby("group", sort=False):
        favorable = float(part["score"].lt(cutoff).mean())
        ratio = None if reference_rate == 0 else float(favorable / reference_rate)
        observed = float(part["target"].mean()) if "target" in part.columns else None
        row = {
            "group": str(name),
            "rows": int(len(part)),
            "score_mean": float(part["score"].mean()),
            "favorable_share": favorable,
            "disparate_impact": ratio,
            "observed_default_rate": observed,
            "reference_group": name == reference_name,
        }
        rows.append(row)
        if (
            name != reference_name
            and len(part) >= min_group_size
            and ratio is not None
            and ratio < FAIRNESS_DISPARATE_IMPACT
        ):
            signals.append(
                {
                    "column": column,
                    "group": str(name),
                    "reference_group": reference_name,
                    "disparate_impact": ratio,
                    "rows": int(len(part)),
                    "meaning": (
                        "Sinal de monitoramento: a massa de score abaixo do corte "
                        "ficou abaixo de quatro quintos do grupo de maior volume. "
                        "Nao e prova de discriminacao."
                    ),
                }
            )
    return {
        "column": column,
        "reference_group": reference_name,
        "groups": sorted(rows, key=lambda item: item["rows"], reverse=True),
        "signals": signals,
    }


def _mitigation() -> list[str]:
    """Lista o que ja foi feito e o que continua sendo limite.

    Returns:
        Controles e limites, sem prometer justica do score.
    """
    return [
        "Sexo, raca, idade, religiao, biometria, CEP e UF nao entram no modelo.",
        "addr_state ficou de fora. Nao volta nesta etapa: o ganho de AUC nao justifica o proxy geografico.",
        "O baseline nao tem limiar. Qualquer corte de concessao e politica, com revisao humana.",
        "class_weight balanced melhora o ranking da classe rara. Nao e tecnica de equidade.",
        "FICO e renda podem carregar vies historico. Excluir a UF nao remove esse residuo.",
        "O recorte e recalculado na referencia e na producao natural. Sinal novo na producao vira item de revisao, nao bloqueio automatico do cliente.",
    ]
