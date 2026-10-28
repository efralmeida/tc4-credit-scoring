"""Prazos de retencao do repositorio e da fintech do cenario.

Nada aqui apaga arquivo. A funcao so declara o que pode existir, por
quanto tempo e por que. Apagar o bruto no meio do curso quebraria a
reproducao da etapa 2.
"""

from typing import Any


def retention_schedule() -> list[dict[str, Any]]:
    """Declara classe, prazo e base de cada artefato.

    Returns:
        Manifesto de retencao, sem caminho de cliente.
    """
    return [
        {
            "artifact": "CSV bruto do Lending Club",
            "path": "data/raw",
            "personal_data": "financeiro identificavel na origem",
            "git": "ignorado",
            "repository_retention": "so enquanto o trabalho precisar ser reproduzido",
            "scenario_retention": "cinco anos apos o fim da relacao de credito, depois apagar ou anonimizar",
            "basis": "Obrigacao legal e exercicio de direitos, arts. 15 e 16 da LGPD",
            "deletion": "apagar o arquivo local; nao ha copia no Git",
        },
        {
            "artifact": "Amostras de referencia e producao",
            "path": "data/processed",
            "personal_data": "minimizado, ainda reidentificavel se cruzado com o bruto",
            "git": "ignorado",
            "repository_retention": "igual ao bruto",
            "scenario_retention": "igual ao bruto",
            "basis": "Execucao de contrato para a amostra de concessao; legitimo interesse para o monitor",
            "deletion": "apagar os CSV locais",
        },
        {
            "artifact": "Modelo e coeficientes agregados",
            "path": "artifacts/models e baseline_coefficients.csv",
            "personal_data": "nenhum dado de linha",
            "git": "versionado",
            "repository_retention": "enquanto o relatorio do trabalho existir",
            "scenario_retention": "vida do modelo mais cinco anos, para explicar decisao passada",
            "basis": "Accountability, art. 6, X, e revisao do art. 20",
            "deletion": "arquivar o binario com a versao; nao reconstruir a base a partir dele",
        },
        {
            "artifact": "Relatorios JSON e HTML agregados",
            "path": "artifacts/reports",
            "personal_data": "agregado; o HTML do Evidently pode embutir amostra ja minimizada",
            "git": "versionado, sem publicacao externa do HTML",
            "repository_retention": "entrega do trabalho",
            "scenario_retention": "cinco anos",
            "basis": "Accountability do monitoramento",
            "deletion": "nao publicar o HTML fora do repositorio; nao extrair linha dele",
        },
        {
            "artifact": "Log de lote",
            "path": "artifacts/logs/batch_health.jsonl",
            "personal_data": "agregado",
            "git": "versionado",
            "repository_retention": "entrega do trabalho",
            "scenario_retention": "cinco anos",
            "basis": "Legitimo interesse do monitoramento",
            "deletion": "substituir na nova execucao; nao ha linha de cliente",
        },
        {
            "artifact": "Log de evento de predicao",
            "path": "artifacts/logs/prediction_events.jsonl",
            "personal_data": "microdado operacional: faixa de score e id aleatorio",
            "git": "ignorado",
            "repository_retention": "noventa dias",
            "scenario_retention": "noventa dias",
            "basis": "Legitimo interesse, pelo teste de balanceamento",
            "deletion": "apagar o JSONL; o id nao permite achar o titular",
        },
        {
            "artifact": "Payload rejeitado pelo contrato",
            "path": "nao persistido",
            "personal_data": "seria dado pessoal, por isso nao e gravado",
            "git": "nao se aplica",
            "repository_retention": "zero",
            "scenario_retention": "zero",
            "basis": "Minimizacao, art. 6, III",
            "deletion": "nao chega a existir como arquivo",
        },
    ]


def retention_manifest() -> dict[str, Any]:
    """Empacota a politica e o que este modulo nao faz.

    Returns:
        Manifesto serializavel.
    """
    return {
        "deletes_files": False,
        "controller": (
            "No cenario, a fintech e a controladora. Neste repositorio, "
            "nao ha tratamento de titular brasileiro."
        ),
        "schedule": retention_schedule(),
    }
