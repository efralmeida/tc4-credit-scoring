# TC4 — Credit Scoring: qualidade, contratos, baseline e drift

Camada de sustentacao de um modelo de credit scoring. A etapa 1 impede que dados invalidos sejam consumidos pelo modelo. A etapa 2 mede se o dado valido ainda parece com o dado de treino.

Etapas 3 e 4 (observabilidade operacional e governanca completa) ainda nao fazem parte deste incremento.

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

## Etapa 2 — drift

Contrato e drift respondem perguntas diferentes. O Pandera barra dado impossivel, como DTI 999. O monitor de drift olha dado possivel cuja distribuicao se afastou do treino.

A referencia continua sendo 2012-2015. A producao natural usa originacoes terminais de 2016-2018, lidas do mesmo arquivo bruto e sem identificadores. Sobre essa amostra, duas variaveis sao alteradas de proposito:

| Variavel | Alteracao | Leitura de negocio |
| --- | --- | --- |
| `fico_range_low` | queda de 40 pontos, limitada a [300, 850] | perfil de credito pior |
| `dti` | multiplicado por 1,35 e somado a 4, limitado a [0, 100] | mais comprometimento de renda |

O clip existe para o drift nao virar falha de contrato. O baseline pontua os tres conjuntos, mas nao recebe `issue_d` nem `loan_status`.

### Evidencia estatistica

| Teste | O que compara | Como ler |
| --- | --- | --- |
| PSI | Proporcao por faixa, numerica ou categorica | Abaixo de 0,10: estavel. De 0,10 a 0,25: moderado. A partir de 0,25: relevante. Nao e p-valor. |
| KS de duas amostras | Distribuicao continua de referencia contra producao | p-valor abaixo de 0,05 rejeita a hipotese de mesma distribuicao. Com dezenas de milhares de linhas, rejeita diferenca pequena. Nao e o KS de credito do baseline. |
| Qui-quadrado | Frequencia das categorias | Equivalente categorico do KS, com a mesma ressalva de amostra grande. |

O alerta operacional usa o PSI. KS e qui-quadrado ficam no JSON como apoio. O Evidently, acima de mil linhas, trocaria o KS pelo Wasserstein; o HTML foi fixado em PSI para a evidencia visual coincidir com o teste de credito.

Ha dois relatorios porque as perguntas sao diferentes:

- `drift_report.html`: referencia contra producao deslocada. E o monitor de producao.
- `drift_injection_report.html`: producao natural contra producao deslocada. Isola FICO e DTI da passagem do tempo.

Concept drift nao e lido na amostra injetada. Ele e a taxa de Charged Off dentro da mesma faixa de score, comparando referencia e producao natural. Mudar FICO e manter o rotulo antigo moveria o score de faixa; isso seria migracao, nao P(Y|X).

```powershell
poetry run python scripts/run_stage2.py
```

`simulate` so gera os CSV. `detect` so gera os relatorios, e exige a simulacao anterior. `--from-reference` nao relê o bruto: reposiciona a referencia em 2016-2018 e deixa explicito que o drift natural da janela real nao aparece.

| Caminho | Conteudo |
| --- | --- |
| `data/processed/production_natural.csv` | Producao 2016-2018, fora do Git. |
| `data/processed/production_drifted.csv` | A mesma amostra, com FICO e DTI deslocados. |
| `artifacts/reports/drift_report.html` | Evidently: referencia contra producao deslocada. |
| `artifacts/reports/drift_injection_report.html` | Evidently: prova das duas variaveis injetadas. |
| `artifacts/reports/drift_summary.json` | PSI, KS, qui-quadrado, scores agregados e concept drift. |

Os HTML nao guardam identificador. O JSON nao guarda score por cliente. Amostra de linha do Evidently, se o HTML a embutir, continua restrita as features ja minimizadas e nao deve ser publicada fora do artefato local.

### Resultado desta execucao

A leitura do bruto encontrou 518.706 emprestimos terminais em 2016-2018. O contrato rejeitou 309 linhas da amostra candidata. A producao ficou com 20.000 linhas.

| Comparacao | O que apareceu |
| --- | --- |
| Injecao, producao natural contra deslocada | So `fico_range_low` (PSI 1,65) e `dti` (PSI 0,85). Nenhuma outra coluna passou de 0,10. |
| Score, referencia contra producao natural | PSI 0,006. A passagem do tempo, sozinha, nao moveu a probabilidade prevista. |
| Score, referencia contra producao deslocada | PSI 0,68. A media do score foi de 0,449 para 0,607. |
| Passagem do tempo nas features | `application_type` ficou moderado (PSI 0,20) e `revol_util` no limite (PSI 0,10). |
| KS em `loan_amnt` | p-valor minusculo com PSI 0,02. Por isso o alerta nao usa so o KS. |

A taxa observada de Charged Off foi de 18,5% na referencia para 22,5% na producao natural. O gap de calibracao, na mesma faixa de score, ficou em 4,7 pontos percentuais. Isso e o sinal de concept drift: o rotulo mudou mais do que o score. A injecao nao altera esse rotulo; ela so empurra o score.

O score medio perto de 0,46 nao e a taxa de inadimplencia. A regressao logistica usa `class_weight="balanced"`, entao a probabilidade nao esta calibrada para a frequencia real. O limiar de concessao continua fora do baseline.
