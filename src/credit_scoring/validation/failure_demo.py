"""Casos sinteticos que devem ser barrados antes do modelo."""

from typing import Any

from credit_scoring.data.schema import (
    CONTRACT_ERRORS,
    SCORING_SCHEMA,
    example_scoring_row,
    scoring_frame,
    summarize_schema_error,
)


def invalid_cases() -> list[dict[str, Any]]:
    """Define falhas controladas, cada uma violando uma regra distinta.

    Returns:
        Casos com nome, payload e colunas que devem aparecer na falha.
    """
    return [
        {
            "name": "dti_sentinela_999",
            "frame": scoring_frame([example_scoring_row(dti=999.0)]),
            "expected_columns": ["dti"],
            "rule": "DTI no intervalo [0, 100]",
        },
        {
            "name": "renda_nao_positiva",
            "frame": scoring_frame([example_scoring_row(annual_inc=0.0)]),
            "expected_columns": ["annual_inc"],
            "rule": "Renda anual estritamente positiva",
        },
        {
            "name": "prazo_fora_do_dominio",
            "frame": scoring_frame([example_scoring_row(term="12 months")]),
            "expected_columns": ["term"],
            "rule": "Dominio fechado de term",
        },
        {
            "name": "fico_nulo",
            "frame": scoring_frame([example_scoring_row(fico_range_low=None)]),
            "expected_columns": ["fico_range_low"],
            "rule": "Completude de FICO",
        },
        {
            "name": "contas_incoerentes",
            "frame": scoring_frame([example_scoring_row(open_acc=20.0, total_acc=5.0)]),
            "expected_columns": [],
            "rule": "total_acc maior ou igual a open_acc",
        },
        {
            "name": "coluna_extra_de_identificacao",
            "frame": scoring_frame([example_scoring_row()]).assign(zip_code="01310"),
            "expected_columns": [],
            "rule": "Schema estrito sem identificadores",
        },
    ]


def demonstrate_controlled_failure() -> dict[str, Any]:
    """Executa os casos invalidos e um controle valido.

    O sucesso desta rotina e a rejeicao de todos os casos invalidos. Se um
    payload incorreto passar, a funcao levanta erro para nao mascarar a falha.

    Returns:
        Relatorio com o resultado de cada caso, sem gravar o payload.

    Raises:
        RuntimeError: Se o contrato aceitar dado invalido ou rejeitar o controle.
    """
    results: list[dict[str, Any]] = []
    for case in invalid_cases():
        results.append(_evaluate_case(case))

    unexpected = [item["name"] for item in results if not item["rejected"]]
    if unexpected:
        raise RuntimeError(f"Contrato aceitou dados invalidos: {unexpected}")

    control_passed = _control_case_passes()
    if not control_passed:
        raise RuntimeError("Contrato rejeitou o caso de controle valido.")

    return {
        "control_valid_passed": control_passed,
        "invalid_cases_rejected": len(results),
        "cases": results,
    }


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Valida um caso e registra apenas o resumo da rejeicao.

    Args:
        case: Definicao produzida por `invalid_cases`.

    Returns:
        Resultado do caso, com indicador de rejeicao.
    """
    try:
        SCORING_SCHEMA.validate(case["frame"], lazy=True)
    except CONTRACT_ERRORS as error:
        summary = summarize_schema_error(error)
        expected = set(case["expected_columns"])
        observed = set(summary["columns"])
        column_match = expected.issubset(observed) if expected else True
        failure_values = error.failure_cases.get("failure_case", [])
        text = " ".join(summary["checks"] + [str(value) for value in failure_values])
        if case["name"] == "coluna_extra_de_identificacao":
            column_match = "zip_code" in text
        if case["name"] == "contas_incoerentes":
            column_match = "total_acc deve ser maior ou igual a open_acc" in text
        return {
            "name": case["name"],
            "rule": case["rule"],
            "rejected": True,
            "expected_signal_found": column_match,
            "columns": summary["columns"],
            "checks": summary["checks"],
        }

    return {
        "name": case["name"],
        "rule": case["rule"],
        "rejected": False,
        "expected_signal_found": False,
        "columns": [],
        "checks": [],
    }


def _control_case_passes() -> bool:
    """Confirma que o contrato nao rejeita uma solicitacao bem formada.

    Returns:
        Verdadeiro quando a linha sintetica valida e aceita.
    """
    try:
        SCORING_SCHEMA.validate(scoring_frame(), lazy=True)
    except CONTRACT_ERRORS:
        return False
    return True


def assert_failure_report(report: dict[str, Any]) -> None:
    """Exige que cada rejeicao tenha o sinal da regra esperada.

    Args:
        report: Saida de `demonstrate_controlled_failure`.

    Raises:
        RuntimeError: Se alguma rejeicao nao apontar a regra prevista.
    """
    missing = [
        item["name"]
        for item in report["cases"]
        if not item["rejected"] or not item["expected_signal_found"]
    ]
    if missing:
        raise RuntimeError(f"Falha controlada incompleta nos casos: {missing}")
