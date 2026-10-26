# TC4 — Credit Scoring: qualidade, contratos e baseline

Camada de sustentacao de um modelo de credit scoring. A etapa 1 impede que dados invalidos sejam consumidos pelo modelo e entrega um baseline treinavel apenas sobre a base de referencia limpa.

Etapas 2 a 4 (drift, observabilidade operacional e governanca completa) ainda nao fazem parte deste incremento.

## Decisao de dados

Fonte local: `data/raw/accepted_2007_to_2018Q4.csv.gz` (Lending Club, classificacao binaria, volume acima de 5.000 registros).

A base de referencia usa emprestimos originados entre janeiro de 2012 e dezembro de 2015 com desfecho terminal (`Fully Paid` ou `Charged Off`). Originacoes a partir de 2016 ficam reservadas para a simulacao de producao da etapa 2.

A amostra final tem ate 80.000 linhas, estratificada por desfecho e reproduzivel com `random_state=42`. A fracao por bloco existe apenas para limitar memoria durante a leitura do arquivo bruto.

Classe positiva: `Charged Off`.

### O que nao entra no modelo

| Motivo | Colunas excluidas |
| --- | --- |
| Minimizacao LGPD | `id`, `member_id`, `emp_title`, `url`, `desc`, `title`, `zip_code`, `addr_state` |
| Vazamento de alvo | `int_rate`, `installment`, `grade`, `sub_grade` e qualquer campo de pagamento, recuperacao ou FICO posterior |
| Metadado, nao feature | `issue_d` permanece na referencia para separar janelas, mas nao entra na inferencia |

`addr_state` fica de fora nesta etapa por minimizacao e por ser um proxy geografico sensivel. A decisao pode ser revisitada na analise de vies da etapa 4, com base legal e teste de necessidade.

Ausencia de `emp_length` vira a categoria explicita `unknown`. Nulo em renda, DTI, FICO ou utilizacao rotativa continua sendo falha de contrato.

## Contrato de dados

O contrato de scoring esta em [src/credit_scoring/data/schema.py](src/credit_scoring/data/schema.py). Ele e estrito: coluna extra, inclusive identificador, e rejeitada.

Regras rigidas destacadas:

1. Renda anual estritamente positiva e ate o teto de sanidade de 1.000.000.
2. DTI no intervalo fechado [0, 100], o que rejeita o sentinela historico 999.
3. `term` restrito a `36 months` e `60 months`.
4. Schema estrito, sem colunas fora do contrato.
5. `total_acc` maior ou igual a `open_acc`.
6. FICO, valor do emprestimo e demais numericos dentro de faixas de sanidade, sem nulos.

Essas faixas separam erro de dados de politica de credito. O baseline nao define limiar de aprovacao. Esse limiar e uma decisao de governanca.

O treino usa um segundo contrato, de referencia. Ele repete as regras de scoring e ainda exige desfecho terminal e `issue_d` dentro de 2012–2015. No treino nao ha quarentena: se a referencia vier adulterada, o ajuste e interrompido. Na preparacao, linhas invalidas sao contadas e descartadas; o payload rejeitado nao e gravado.

## Modelo baseline

Regressao logistica do scikit-learn, com `class_weight="balanced"`, padronizacao das variaveis numericas e one-hot das categoricas. A entrada passa pelo contrato antes de `predict_proba`.

Metricas de holdout estratificado (20%): ROC-AUC, average precision, KS de credito e Brier. O KS aqui e a maior distancia absoluta entre as curvas acumuladas de verdadeiros e falsos positivos. Nao e um p-valor e nao substitui o teste de drift da etapa 2.

Resultado obtido nesta execucao, sem `grade` e sem `int_rate`:

| Metrica | Valor |
| --- | --- |
| ROC-AUC | 0,697 |
| Average precision | 0,347 |
| KS | 0,290 |
| Brier | 0,219 |
| Taxa de Charged Off no teste | 18,5% |

A quarentena da preparacao leu 2.260.701 linhas, encontrou 786.818 emprestimos terminais na janela e rejeitou 1.611 linhas da amostra candidata antes de fixar a referencia em 80.000 registros (65.232 Fully Paid e 14.768 Charged Off).

## Como executar

```powershell
poetry install
poetry run python scripts/run_stage1.py
```

Comandos parciais: `prepare`, `train` e `demonstrate-failure`.

```powershell
poetry run pytest
```

Artefatos gerados:

| Caminho | Conteudo |
| --- | --- |
| `data/processed/reference_sample.csv` | Amostra de referencia. Fica fora do Git porque `data/` esta ignorado. |
| `artifacts/models/credit_baseline.joblib` | Pipeline treinado e metadados. |
| `artifacts/reports/baseline_metrics.json` | Metricas de holdout. |
| `artifacts/reports/baseline_coefficients.csv` | Coeficientes agregados, sem dados de cliente. |
| `artifacts/reports/preparation_report.json` | Contagens da quarentena. |
| `artifacts/reports/contract_failure_report.json` | Prova da falha controlada. |

A falha controlada cobre DTI 999, renda nao positiva, prazo fora do dominio, FICO nulo, contas incoerentes e uma coluna `zip_code`. Um caso valido precisa passar, para mostrar que o contrato nao rejeita tudo.

## Privacidade nesta etapa

Nao ha identificador direto no dataset de modelagem, no modelo ou nos relatorios. Relatorios de rejeicao guardam nomes de regras e contagens, nao as linhas recusadas. A base bruta permanece apenas em `data/raw`.

A base legal da decisao de credito, o prazo de retencao e a mitigacao de vies serao documentados na etapa 4. Esta etapa ja aplica minimizacao e finalidade: so entram variaveis disponiveis na concessao e necessarias ao baseline.
