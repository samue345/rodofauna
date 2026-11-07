# RodoFauna ML: classificação, regressão e redes neurais

## Resumo

Este trabalho desenvolve um pipeline de aprendizado supervisionado para analisar
registros de atropelamentos de fauna no Distrito Federal. Foram avaliados
modelos tradicionais e redes neurais MLP em problemas de classificação
multiclasse e regressão.

Após a inspeção da planilha original, foi identificado que os 10.328 registros
estavam duplicados em dois blocos. As cópias eram iguais, exceto pelo campo
técnico `OBJECTID`. O arquivo original foi preservado e o tratamento foi feito
durante o carregamento: `OBJECTID` foi removido e as duplicatas foram
eliminadas. A base final possui 5.164 ocorrências únicas.

## 1. Dados e análise exploratória

Os dados são provenientes do Programa RodoFauna do IBRAM/DF. A base final
possui 5.164 registros e 19 atributos após a remoção de `OBJECTID`.

Foram mantidos apenas os registros de fauna silvestre para os modelos. Isso
resultou em 4.422 registros silvestres e 742 registros domésticos.

Entre os registros silvestres, as classes taxonômicas são:

| Classe | Registros |
|---|---:|
| Aves | 3.009 |
| Répteis | 690 |
| Mamíferos | 448 |
| Anfíbios | 274 |
| Não identificado | 1 |

O registro `nao_identificado` foi removido da classificação multiclasse. Não
foram encontrados valores ausentes ou coordenadas inválidas na base utilizada.
O desbalanceamento é relevante, pois aves representam a maioria das
ocorrências.

## 2. Pré-processamento

As variáveis numéricas foram padronizadas com `StandardScaler`:

- ano;
- coordenada X;
- coordenada Y.

As variáveis categóricas foram transformadas com One-Hot Encoding:

- mês;
- dia da semana;
- unidade de conservação.

O pré-processador é ajustado somente no conjunto de treinamento. Validação e
teste usam as regras aprendidas no treinamento, evitando vazamento de
informação.

Os dados foram divididos em treinamento, validação e teste, mantendo o teste
separado durante a escolha dos modelos e hiperparâmetros.

O campo `data_dt` foi usado para criar atributos temporais adicionais, sem ser
passado diretamente como texto para os modelos:

- mês numérico;
- dia do ano;
- indicador de fim de semana;
- estação do ano;
- período chuvoso ou seco.

## 3. Classificação multiclasse com MLP

O alvo foi a classe taxonômica do animal. A MLP inicial possui duas camadas
ocultas com 64 e 32 neurônios, ativação ReLU e otimizador Adam.

Os modelos foram comparados usando a mesma divisão dos dados:

| Modelo | F1 Macro validação | F1 Macro teste | Acurácia teste |
|---|---:|---:|---:|
| Regressão Logística | 0,2024 | 0,2075 | 0,6822 |
| Random Forest | 0,4253 | 0,4379 | 0,5678 |
| MLP | 0,3219 | 0,3499 | 0,6913 |

O Random Forest obteve o maior F1 Macro no teste. A acurácia da MLP foi maior
que seu F1 Macro porque a classe aves é dominante. Por isso, a acurácia
isolada não é suficiente para avaliar o desempenho.

As curvas de loss e acurácia estão em:

`reports/generated/mlp_training_curves.png`

## 4. Regressão com MLP

Para a regressão, os registros foram agregados por célula espacial de 500 metros
e mês. A variável alvo foi o número de atropelamentos em cada célula-mês.

Foram comparados Regressão Linear, Random Forest Regressor e MLPRegressor.

| Modelo | MAE teste | RMSE teste | R² teste |
|---|---:|---:|---:|
| Regressão Linear | 0,630 | 0,935 | 0,046 |
| Random Forest Regressor | 0,640 | 1,040 | -0,181 |
| MLPRegressor | 0,632 | 0,951 | 0,012 |

A Regressão Linear apresentou o melhor resultado nesta primeira configuração.
Os valores baixos de R² indicam que os atributos disponíveis explicam pouco da
variação da quantidade de atropelamentos.

Uma limitação importante é que foram agregadas apenas células-mês com registros
observados. Células sem registros não foram automaticamente consideradas como
risco zero.

## 5. Otimização com Optuna

O Optuna foi utilizado para otimizar a MLP de classificação. A função objetivo
foi maximizar o F1 Macro na validação. O conjunto de teste não foi usado para
selecionar hiperparâmetros.

Foram executados 50 trials. A melhor configuração foi:

```text
camadas ocultas: 32 e 64 neurônios
ativação: relu
alpha: 0,000271
learning rate: 0,001300
batch size: 32
```

O melhor F1 Macro de validação foi 0,3059, enquanto o F1 Macro no teste foi
0,3477. A busca ampliada mostrou que a arquitetura inicial `(64, 32)` não é
necessariamente a melhor: neste experimento, `(32, 64)` obteve melhor resultado
na validação. Ainda assim, a seleção deve ser interpretada com cautela porque
o teste permanece reservado para a avaliação final.
melhorar a métrica de validação em uma execução não garante ganho no conjunto
de teste.

Resultados detalhados:

`reports/generated/classification_phase5_optuna.json`

## 6. Regularização e overfitting

Foram comparadas duas MLPs com a arquitetura encontrada pelo Optuna. A fase de
regularização também testou diferentes valores de `alpha` usando o mesmo
treinamento iterativo com `partial_fit`:

- sem regularização: `alpha=0.0`;
- com regularização L2: melhor `alpha` na validação entre os valores testados.

| Modelo | F1 Macro validação | F1 Macro teste | Acurácia teste |
|---|---:|---:|---:|
| Sem regularização | 0,3388 | 0,3831 | 0,6807 |
| Com L2 (`alpha=0,000271`) | 0,3468 | 0,3853 | 0,6837 |

Nesta configuração, a regularização L2 aumentou o F1 Macro na validação, mas
também apresentou uma pequena melhora no conjunto de teste. Isso mostra a
importância de selecionar o hiperparâmetro usando apenas a validação e reservar
o teste para a avaliação final. As curvas permitem observar a diferença entre
o comportamento no treinamento e na validação e devem ser utilizadas para
discutir overfitting.

Resultados e curvas:

`reports/generated/classification_phase6_regularization.json`

## 7. Discussão final

### 7.1 Atributos mais relacionados

As coordenadas X e Y, a unidade de conservação e as variáveis temporais foram
usadas como indicadores espaciais e temporais. Entretanto, os resultados não
permitem afirmar causalidade. A base registra ocorrências, não mede diretamente
risco de atropelamento.

### 7.2 MLP versus modelos tradicionais

Na classificação, o Random Forest apresentou o melhor F1 Macro no teste. A MLP
teve desempenho intermediário e a Regressão Logística foi a pior neste
experimento.

Na regressão, a Regressão Linear apresentou o melhor resultado. Isso indica que
a maior complexidade da MLP não produziu automaticamente melhor generalização.

### 7.3 Desbalanceamento

A predominância de aves favorece a acurácia geral. As métricas macro são mais
adequadas porque dão o mesmo peso às classes menores, como anfíbios.

### 7.4 Limitações

As principais limitações são:

- ausência de amostras confiáveis de locais sem atropelamento;
- possível viés de esforço amostral nas áreas percorridas;
- poucos atributos ambientais e rodoviários;
- desbalanceamento entre classes;
- uso de divisão aleatória, que pode colocar pontos próximos em conjuntos
  diferentes;
- regressão baseada somente em células-mês com ocorrências observadas.

### 7.5 Melhorias futuras

O modelo poderia ser aprimorado com:

- precipitação e temperatura;
- uso e cobertura do solo;
- distância a corpos d'água;
- características da rodovia;
- volume de tráfego;
- velocidade máxima permitida;
- divisão temporal ou espacial mais rigorosa;
- construção documentada de células com zero ocorrências.

## Conclusão

Foi construído um pipeline completo de preparação, classificação, regressão,
otimização e regularização. Os resultados mostram que modelos mais complexos
não foram necessariamente superiores aos modelos tradicionais. A qualidade da
definição da unidade de análise, a representatividade dos dados e o tratamento
do desbalanceamento são tão importantes quanto a escolha da arquitetura da
rede neural.

Os experimentos podem ser reproduzidos executando:

```bash
python -m src.main
```
