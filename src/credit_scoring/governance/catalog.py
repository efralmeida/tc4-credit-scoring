"""Classificacao dos dados e bases legais do cenario de credito.

O arquivo do Lending Club e publico e americano. Nao ha titular
brasileiro neste repositorio. O catalogo descreve o controle que a
fintech do enunciado precisaria se o mesmo desenho tratasse dado de
pessoa no Brasil. Nao declara a FIAP, nem o aluno, controlador dessa base.
"""

from typing import Any

from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    EXCLUDED_IDENTIFIERS,
    EXCLUDED_LEAKAGE,
    NUMERIC_FEATURES,
)

_FINANCIAL = (
    "Dado pessoal financeiro quando ligado a uma pessoa. Na tabela de "
    "modelagem nao ha identificador direto, mas o cruzamento com o bruto "
    "reidentificaria. Por isso o bruto fica fora do Git e fora do log."
)
_PROXY = (
    "Nao e dado pessoal sensivel. Pode funcionar como proxy socioeconomico "
    "e entra no monitoramento de vies, nao como atributo protegido."
)


def data_catalog() -> list[dict[str, str]]:
    """Lista o papel de cada coluna no tratamento.

    Returns:
        Catalogo com classificacao LGPD e a decisao de uso.
    """
    rows: list[dict[str, str]] = []
    for column in EXCLUDED_IDENTIFIERS:
        rows.append(
            {
                "column": column,
                "class": "identificador ou proxy direto",
                "lgpd": "dado pessoal",
                "sensitive_personal_data": "nao",
                "decision": "excluido na leitura",
                "reason": "Minimizacao. Nao e necessario ao baseline e aumenta reidentificacao.",
            }
        )
    for column in EXCLUDED_LEAKAGE:
        rows.append(
            {
                "column": column,
                "class": "vazamento ou pos-concessao",
                "lgpd": "dado pessoal financeiro, se ligado ao titular",
                "sensitive_personal_data": "nao",
                "decision": "excluido do modelo",
                "reason": "Nao existe no momento da concessao, ou vaza o desfecho.",
            }
        )
    for column in NUMERIC_FEATURES:
        rows.append(
            {
                "column": column,
                "class": "feature de concessao",
                "lgpd": "dado pessoal financeiro",
                "sensitive_personal_data": "nao",
                "decision": "entra no baseline",
                "reason": _FINANCIAL,
            }
        )
    for column in CATEGORICAL_FEATURES:
        rows.append(
            {
                "column": column,
                "class": "feature categorica",
                "lgpd": "dado pessoal",
                "sensitive_personal_data": "nao",
                "decision": "entra no baseline e no recorte de vies",
                "reason": _PROXY,
            }
        )
    rows.append(
        {
            "column": "purpose=medical",
            "class": "finalidade declarada do emprestimo",
            "lgpd": "nao e dado de saude",
            "sensitive_personal_data": "nao",
            "decision": "categoria de purpose, nao dado sensivel",
            "reason": (
                "Art. 5, II, da LGPD fala de dado de saude da pessoa. "
                "A categoria medical diz para que o dinheiro foi pedido, nao um diagnostico."
            ),
        }
    )
    return rows


def legal_bases() -> list[dict[str, str]]:
    """Registra a base legal de cada finalidade, no cenario da fintech.

    Returns:
        Finalidade, base e o limite dessa base.
    """
    return [
        {
            "purpose": "Analisar a proposta de credito",
            "basis": "Execucao de contrato ou de procedimentos preliminares, art. 7, V, da LGPD",
            "limit": "So features disponiveis na concessao. Nao cobre monitoramento historico nem marketing.",
        },
        {
            "purpose": "Atender dever de informacao e prevencao a fraude no credito",
            "basis": "Obrigacao legal, art. 7, II, da LGPD, com o CDC, art. 43",
            "limit": (
                "Este baseline nao consulta biro e nao grava resultado em cadastro positivo. "
                "A Lei 12.414/2011 so entraria se houvesse esse cadastro."
            ),
        },
        {
            "purpose": "Monitorar qualidade, drift e saude do modelo",
            "basis": "Legitimo interesse, art. 7, IX, da LGPD",
            "limit": "Agregados e faixas de score. Nao e base para conceder ou negar o credito do titular.",
        },
    ]


def legitimate_interest_test() -> dict[str, str]:
    """Documenta o teste de balanceamento do monitoramento.

    Returns:
        Finalidade, necessidade, balanceamento e conclusao.
    """
    return {
        "purpose": "Detectar dado invalido e populacao diferente da de treino.",
        "necessity": "Sem agregado de volume, erro, latencia e PSI, a degradacao passa em silencio.",
        "balance": (
            "O log nao leva identificador, nao leva feature e nao leva a linha rejeitada. "
            "O evento de predicao guarda faixa de score e um id aleatorio. "
            "O acesso fica com quem opera o pipeline."
        ),
        "conclusion": (
            "O legitimo interesse sustenta o monitoramento. Nao sustenta a decisao de credito, "
            "que depende do art. 7, V, e, se for automatizada, do art. 20."
        ),
    }


def automated_decision_policy() -> dict[str, str]:
    """Explica por que o baseline, sozinho, nao e decisao automatizada.

    Returns:
        Politica de revisao e explicacao.
    """
    return {
        "article": "Art. 20 da LGPD e art. 43 do CDC",
        "current_behavior": (
            "O artefato grava decision_threshold nulo. O score ordena risco. "
            "Nao aprova nem nega. class_weight balanced tira a leitura do score como probabilidade real."
        ),
        "if_used_in_production": (
            "A concessao precisaria de criterio escrito, revisao humana pedida pelo titular "
            "e explicacao da decisao. Coeficiente agregado explica o modelo, nao o caso concreto."
        ),
        "rights": (
            "Acesso, correcao, informacao sobre compartilhamento e revisao da decisao. "
            "Eliminacao quando a retencao nao tiver mais base. O canal e o encarregado "
            "indicado no registro de operacoes da fintech."
        ),
    }


def privacy_principles() -> dict[str, Any]:
    """Reune o enquadramento usado no README e no HTML.

    Returns:
        Cenario, catalogo, bases e politica de decisao.
    """
    return {
        "scenario": (
            "Fintech brasileira de credito. O dataset didatico e o Lending Club, "
            "publico e americano. O mapeamento LGPD nao cria titular brasileiro neste arquivo."
        ),
        "sensitive_personal_data_in_model": False,
        "catalog": data_catalog(),
        "legal_bases": legal_bases(),
        "legitimate_interest_test": legitimate_interest_test(),
        "automated_decision": automated_decision_policy(),
    }
