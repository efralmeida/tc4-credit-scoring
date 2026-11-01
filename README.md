# TC4 — Credit Scoring: qualidade, drift, observabilidade e governança

Camada de sustentação de um modelo de credit scoring. A etapa 1 impede que dados inválidos sejam consumidos pelo modelo. A etapa 2 mede se o dado válido ainda parece com o dado de treino. A etapa 3 transforma isso em log e indicador de saúde. A etapa 4 diz o que pode ser tratado, por quanto tempo, e o que daqueles alertas é causa.

O dataset didático é o Lending Club, público e americano. O mapeamento de LGPD descreve o controle que a fintech do enunciado precisaria. Não cria titular brasileiro neste arquivo e não torna o aluno controlador dessa base.

## Decisão de dados

Fonte local: `data/raw/accepted_2007_to_2018Q4.csv.gz` (Lending Club, classificação binária, volume acima de 5.000 registros).

A base de referência usa empréstimos originados entre janeiro de 2012 e dezembro de 2015 com desfecho terminal (`Fully Paid` ou `Charged Off`). Originações a partir de 2016 ficam reservadas para a simulação de produção da etapa 2.

A amostra final tem até 80.000 linhas, estratificada por desfecho e reproduzível com `random_state=42`. A fração por bloco existe apenas para limitar memória durante a leitura do arquivo bruto.

Classe positiva: `Charged Off`.

### O que não entra no modelo

| Motivo | Colunas excluídas |
| --- | --- |
| Minimização LGPD | `id`, `member_id`, `emp_title`, `url`, `desc`, `title`, `zip_code`, `addr_state` |
| Vazamento de alvo | `int_rate`, `installment`, `grade`, `sub_grade` e qualquer campo de pagamento, recuperação ou FICO posterior |
| Metadado, não feature | `issue_d` permanece na referência para separar janelas, mas não entra na inferência |

`addr_state` fica de fora por minimização e por ser um proxy geográfico. A etapa 4 não o traz de volta: o ganho eventual de AUC não justifica o proxy.

Ausência de `emp_length` vira a categoria explícita `unknown`. Nulo em renda, DTI, FICO ou utilização rotativa continua sendo falha de contrato.

## Contrato de dados

O contrato de scoring está em [src/credit_scoring/data/schema.py](src/credit_scoring/data/schema.py). Ele é estrito: coluna extra, inclusive identificador, é rejeitada.

Regras rígidas destacadas:

1. Renda anual estritamente positiva e até o teto de sanidade de 1.000.000.
2. DTI no intervalo fechado [0, 100], o que rejeita o sentinela histórico 999.
3. `term` restrito a `36 months` e `60 months`.
4. Schema estrito, sem colunas fora do contrato.
5. `total_acc` maior ou igual a `open_acc`.
6. FICO, valor do empréstimo e demais numéricos dentro de faixas de sanidade, sem nulos.

Essas faixas separam erro de dados de política de crédito. O baseline não define limiar de aprovação. Esse limiar é uma decisão de governança.

O treino usa um segundo contrato, de referência. Ele repete as regras de scoring e ainda exige desfecho terminal e `issue_d` dentro de 2012–2015. No treino não há quarentena: se a referência vier adulterada, o ajuste é interrompido. Na preparação, linhas inválidas são contadas e descartadas; o payload rejeitado não é gravado.

## Modelo baseline

Regressão logística do scikit-learn, com `class_weight="balanced"`, padronização das variáveis numéricas e one-hot das categóricas. A entrada passa pelo contrato antes de `predict_proba`.

Métricas de holdout estratificado (20%): ROC-AUC, average precision, KS de crédito e Brier. O KS aqui é a maior distância absoluta entre as curvas acumuladas de verdadeiros e falsos positivos. Não é um p-valor e não substitui o teste de drift da etapa 2.

Resultado obtido nesta execução, sem `grade` e sem `int_rate`:

| Métrica | Valor |
| --- | --- |
| ROC-AUC | 0,697 |
| Average precision | 0,347 |
| KS | 0,290 |
| Brier | 0,219 |
| Taxa de Charged Off no teste | 18,5% |

A quarentena da preparação leu 2.260.701 linhas, encontrou 786.818 empréstimos terminais na janela e rejeitou 1.611 linhas da amostra candidata antes de fixar a referência em 80.000 registros (65.232 Fully Paid e 14.768 Charged Off).

## Como executar

```powershell
poetry install
poetry run python scripts/run_stage1.py
poetry run python scripts/run_stage2.py
poetry run python scripts/run_stage3.py
poetry run python scripts/run_stage4.py
poetry run pytest
```

Comandos parciais da etapa 1: `prepare`, `train` e `demonstrate-failure`. Na etapa 2, `simulate` e `detect`.

Artefatos gerados:

| Caminho | Conteúdo |
| --- | --- |
| `data/processed/reference_sample.csv` | Amostra de referência. Fica fora do Git porque `data/` está ignorado. |
| `artifacts/models/credit_baseline.joblib` | Pipeline treinado e metadados. |
| `artifacts/reports/baseline_metrics.json` | Métricas de holdout. |
| `artifacts/reports/baseline_coefficients.csv` | Coeficientes agregados, sem dados de cliente. |
| `artifacts/reports/preparation_report.json` | Contagens da quarentena. |
| `artifacts/reports/contract_failure_report.json` | Prova da falha controlada. |

A falha controlada cobre DTI 999, renda não positiva, prazo fora do domínio, FICO nulo, contas incoerentes e uma coluna `zip_code`. Um caso válido precisa passar, para mostrar que o contrato não rejeita tudo.

## Privacidade desde a etapa 1

Não há identificador direto no dataset de modelagem, no modelo ou nos relatórios. Relatórios de rejeição guardam nomes de regras e contagens, não as linhas recusadas. A base bruta permanece apenas em `data/raw`. A base legal, o prazo e o viés estão na etapa 4.

## Etapa 2 — drift

Contrato e drift respondem perguntas diferentes. O Pandera barra dado impossível, como DTI 999. O monitor de drift olha dado possível cuja distribuição se afastou do treino.

A referência continua sendo 2012-2015. A produção natural usa originações terminais de 2016-2018, lidas do mesmo arquivo bruto e sem identificadores. Sobre essa amostra, duas variáveis são alteradas de propósito:

| Variável | Alteração | Leitura de negócio |
| --- | --- | --- |
| `fico_range_low` | queda de 40 pontos, limitada a [300, 850] | perfil de crédito pior |
| `dti` | multiplicado por 1,35 e somado a 4, limitado a [0, 100] | mais comprometimento de renda |

O clip existe para o drift não virar falha de contrato. O baseline pontua os três conjuntos, mas não recebe `issue_d` nem `loan_status`.

### Evidência estatística

| Teste | O que compara | Como ler |
| --- | --- | --- |
| PSI | Proporção por faixa, numérica ou categórica | Abaixo de 0,10: estável. De 0,10 a 0,25: moderado. A partir de 0,25: relevante. Não é p-valor. |
| KS de duas amostras | Distribuição contínua de referência contra produção | p-valor abaixo de 0,05 rejeita a hipótese de mesma distribuição. Com dezenas de milhares de linhas, rejeita diferença pequena. Não é o KS de crédito do baseline. |
| Qui-quadrado | Frequência das categorias | Equivalente categórico do KS, com a mesma ressalva de amostra grande. |

O alerta operacional usa o PSI. KS e qui-quadrado ficam no JSON como apoio. O Evidently, acima de mil linhas, trocaria o KS pelo Wasserstein; o HTML foi fixado em PSI para a evidência visual coincidir com o teste de crédito.

Há dois relatórios porque as perguntas são diferentes:

- `drift_report.html`: referência contra produção deslocada. É o monitor de produção.
- `drift_injection_report.html`: produção natural contra produção deslocada. Isola FICO e DTI da passagem do tempo.

Concept drift não é lido na amostra injetada. Ele é a taxa de Charged Off dentro da mesma faixa de score, comparando referência e produção natural. Mudar FICO e manter o rótulo antigo moveria o score de faixa; isso seria migração, não P(Y|X).

```powershell
poetry run python scripts/run_stage2.py
```

`simulate` só gera os CSV. `detect` só gera os relatórios, e exige a simulação anterior. `--from-reference` não relê o bruto: reposiciona a referência em 2016-2018 e deixa explícito que o drift natural da janela real não aparece.

| Caminho | Conteúdo |
| --- | --- |
| `data/processed/production_natural.csv` | Produção 2016-2018, fora do Git. |
| `data/processed/production_drifted.csv` | A mesma amostra, com FICO e DTI deslocados. |
| `artifacts/reports/drift_report.html` | Evidently: referência contra produção deslocada. |
| `artifacts/reports/drift_injection_report.html` | Evidently: prova das duas variáveis injetadas. |
| `artifacts/reports/drift_summary.json` | PSI, KS, qui-quadrado, scores agregados e concept drift. |
| `artifacts/logs/batch_health.jsonl` | Log agregado de lote: volume, nulos, latência, erro e PSI. |
| `artifacts/logs/prediction_events.jsonl` | Evento de predição, só com faixa de score. Fora do Git. |
| `artifacts/reports/health_summary.json` | Indicadores de saúde e drift confirmatório. |
| `artifacts/reports/observability_report.html` | Dashboard estático da etapa 3. |
| `artifacts/reports/governance_report.json` | Catálogo, retenção, viés e análise causal. |
| `artifacts/reports/governance_report.html` | Página para mostrar a etapa 4. |

Os HTML não guardam identificador. O JSON não guarda score por cliente. Amostra de linha do Evidently, se o HTML a embutir, continua restrita às features já minimizadas e não deve ser publicada fora do artefato local.

### Resultado desta execução

A leitura do bruto encontrou 518.706 empréstimos terminais em 2016-2018. O contrato rejeitou 309 linhas da amostra candidata. A produção ficou com 20.000 linhas.

| Comparação | O que apareceu |
| --- | --- |
| Injeção, produção natural contra deslocada | Só `fico_range_low` (PSI 1,65) e `dti` (PSI 0,85). Nenhuma outra coluna passou de 0,10. |
| Score, referência contra produção natural | PSI 0,006. A passagem do tempo, sozinha, não moveu a probabilidade prevista. |
| Score, referência contra produção deslocada | PSI 0,68. A média do score foi de 0,449 para 0,607. |
| Passagem do tempo nas features | `application_type` ficou moderado (PSI 0,20) e `revol_util` no limite (PSI 0,10). |
| KS em `loan_amnt` | p-valor minúsculo com PSI 0,02. Por isso o alerta não usa só o KS. |

A taxa observada de Charged Off foi de 18,5% na referência para 22,5% na produção natural. O gap de calibração, na mesma faixa de score, ficou em 4,7 pontos percentuais. Isso é o sinal de concept drift: o rótulo mudou mais do que o score. A injeção não altera esse rótulo; ela só empurra o score.

O score médio perto de 0,46 não é a taxa de inadimplência. A regressão logística usa `class_weight="balanced"`, então a probabilidade não está calibrada para a frequência real. O limiar de concessão continua fora do baseline.


## Etapa 3 — observabilidade

O monitor não é uma API. A produção natural entra em lotes mensais, de 2016-01 a 2018-12. Cada lote passa de novo pelo contrato, é cronometrado e só então recebe score. Dois lotes extras não são clientes novos: um probe com dado inválido e um incidente com o FICO e o DTI já deslocados.

O que se mede:

| Indicador | Regra | Nesta execução |
| --- | --- | --- |
| Volume de predições | Linhas aceitas | 20.000 na janela real; 28 no probe; 2.000 no incidente |
| Taxa de erro | Rejeitadas / recebidas | 12 / 22.040, ou 0,054%. O payload rejeitado não é gravado |
| Percentual de nulos | Células vazias nas colunas do contrato, antes da rejeição | 0,0019% |
| Latência média | `predict_proba` do lote aceito, dividido pelas linhas | 0,135 ms por linha. O alerta só paginaria acima de 50 ms |
| Alerta de drift | PSI de FICO, DTI ou score ≥ 0,25, com pelo menos 200 linhas | Dois lotes paginam. Só o incidente se confirma |

A evidência do alerta continua sendo o PSI, o mesmo da etapa 2. Abaixo de 0,10, estável. De 0,10 a 0,25, moderado. A partir de 0,25, relevante. Não é p-valor. O KS não pagina: na amostra inteira ele rejeita até `loan_amnt`, cujo PSI é 0,02.

O lote de 2018-03, com 260 linhas, paginou `fico_range_low` com PSI 0,257. Na amostra inteira o PSI dessa coluna é 0,019. O dashboard marca isso como página não confirmada. O incidente, com as duas variáveis deslocadas, confirma. O PSI do score na amostra natural permanece 0,006. Na amostra deslocada ele sai da faixa estável, como na etapa 2.

O log de lote é agregado e fica no repositório. O log de evento guarda faixa de score, de 0,2 em 0,2, e um UUID que não é chave de cliente. A probabilidade exata não entra, para não servir de quase-identificador. Esse arquivo está no `.gitignore`. Uma amostra de cinco eventos, sem feature, fica em `artifacts/logs/prediction_events_sample.jsonl`.

```powershell
poetry run python scripts/run_stage3.py
```

Abrir `artifacts/reports/observability_report.html`.

## Etapa 4 — governança, LGPD e análise causal

### O que é dado pessoal aqui

Identificador direto (`id`, `member_id`, `emp_title`, `url`, `desc`, `title`, `zip_code`, `addr_state`) é excluído na leitura. As features que ficam são dado pessoal financeiro quando ligadas a uma pessoa. Na tabela de modelagem não há identificador, mas o cruzamento com o bruto reidentificaria. Por isso o bruto e os CSV processados ficam fora do Git.

Não há dado pessoal sensível no modelo. Sexo, raça, religião, saúde, genética e biometria não entram. `purpose = medical` é a finalidade declarada do empréstimo, não um diagnóstico. O art. 5º, II, da LGPD não se aplica a essa categoria.

### Base legal, no cenário da fintech

| Finalidade | Base | Limite |
| --- | --- | --- |
| Analisar a proposta | Execução de contrato ou de procedimentos preliminares, art. 7º, V | Só o que existe na concessão |
| Dever de informação no crédito | Obrigação legal, art. 7º, II, com o art. 43 do CDC | Este baseline não consulta birô e não grava cadastro positivo. A Lei 12.414/2011 só entraria se houvesse esse cadastro |
| Monitorar drift e qualidade | Legítimo interesse, art. 7º, IX | Agregados e faixa de score. Não serve para conceder ou negar o crédito da pessoa |

O teste de balanceamento do monitoramento: a finalidade é ver degradação silenciosa; o meio é necessário porque sem volume, erro, latência e PSI ninguém vê o problema; o equilíbrio vem de não gravar identificador, feature nem linha rejeitada. Conclusão: o legítimo interesse sustenta o monitor. Não sustenta a decisão de crédito.

O artefato grava `decision_threshold` nulo. O score ordena risco. Não aprova. Se um dia a fintech usar o score para decidir, o art. 20 da LGPD exige revisão humana pedida pelo titular e explicação da decisão. Coeficiente agregado explica o modelo, não o caso concreto. `class_weight="balanced"` também impede ler o score como probabilidade real de inadimplência.

Direitos no cenário: acesso, correção, informação sobre compartilhamento, revisão da decisão automatizada e eliminação quando a retenção perder a base. O canal é o encarregado indicado no registro de operações da fintech. Este repositório não publica e-mail fictício de DPO.

### Retenção

A política executável está em `src/credit_scoring/governance/retention.py`. O módulo não apaga arquivo.

| Artefato | Git | Prazo no cenário |
| --- | --- | --- |
| CSV bruto e amostras | Ignorado | Cinco anos após o fim da relação de crédito, depois apagar ou anonimizar. Neste repositório, só enquanto a reprodução do trabalho exigir |
| Modelo e coeficientes | Versionado | Vida do modelo mais cinco anos |
| Relatórios agregados | Versionado, sem publicar o HTML fora daqui | Cinco anos. O HTML do Evidently pode embutir amostra já minimizada |
| Log de lote | Versionado | Cinco anos |
| Log de evento | Ignorado | Noventa dias |
| Payload rejeitado | Não existe | Zero |

### Viés

O recorte compara a massa de score abaixo de 0,50. Esse corte não é aprovação. É só a régua para ver se um grupo fica com menos score baixo do que o grupo de maior volume. A regra dos quatro quintos entra como sinal de monitoramento, não como laudo de discriminação.

Na produção natural, seis recortes ficaram abaixo de 0,80, todos com pelo menos 200 linhas:

| Recorte | Grupo | Referência | Razão |
| --- | --- | --- | --- |
| `term` | 60 months | 36 months | 0,19 |
| `emp_length` | unknown | 10+ years | 0,64 |
| `home_ownership` | RENT | MORTGAGE | 0,74 |
| `verification_status` | Verified | Source Verified | 0,78 |
| `purpose` | small_business | debt_consolidation | 0,46 |
| `application_type` | Joint App | Individual | 0,79 |

O prazo de 60 meses é feature de produto. O modelo foi treinado para dar mais risco a esse prazo. O sinal não prova discriminação. Prova outra coisa: se alguém transformar 0,50 em recusa automática, o prazo longo concentra a recusa. Por isso o limiar continua fora do baseline.

O que já foi feito: atributos protegidos e proxy geográfico fora; linha rejeitada não gravada; recorte repetido na referência e na produção; sinal vira revisão de política, não bloqueio do cliente. O que não foi removido: FICO e renda podem carregar viés histórico. `class_weight="balanced"` não é técnica de equidade.

### Análise causal

Há dois contrastes. Eles não respondem à mesma pergunta.

A injeção é intervenção nas mesmas linhas. FICO cai 40 pontos e o DTI sobe. O rótulo não muda. O score médio vai de 0,449 para 0,607 e o PSI do score, natural contra deslocada, fica em 0,79. Isso causa o prediction drift do incidente. Não é concept drift. Também não explica 2016–2018.

A janela real é observacional. A taxa de Charged Off vai de 18,46% para 22,46%, diferença de 3,995 pontos. A decomposição de Kitagawa, nas mesmas faixas de score da etapa 2, separa:

| Componente | Valor | Leitura |
| --- | --- | --- |
| Efeito de composição | −0,674 ponto | A mistura de score não empurrou o default para cima. Até reduziu um pouco |
| Efeito de taxa | +4,669 pontos | Dentro da mesma faixa, o default subiu. É o sinal de concept drift |
| Resíduo | ~0 | A conta fecha. Não sobra um terceiro termo escondido |

O mix de `application_type` não explica. Com o mix de 2012–2015 aplicado às taxas de 2016–2018, o gap que sobra é 3,94 pontos, perto dos 4,00 observados.

O que não está identificado, e por isso não entra como causa: cenário de crédito mais duro, mudança de política do Lending Club que estas colunas não veem, e o PSI moderado de `revol_util`. PSI mostra associação. A decomposição localiza o efeito. Nenhuma das duas nomeia o fator externo.

```powershell
poetry run python scripts/run_stage4.py
```

Abrir `artifacts/reports/governance_report.html` e `artifacts/reports/governance_report.json`.