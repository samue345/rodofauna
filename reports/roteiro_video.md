# Roteiro do vídeo - RodoFauna ML

## 0:00 - 0:50 | Apresentação do problema

Olá. Neste trabalho analisamos registros de atropelamentos de fauna silvestre
no Distrito Federal, utilizando dados do Programa RodoFauna do IBRAM.

O objetivo foi construir um pipeline de aprendizado supervisionado envolvendo
classificação, regressão, redes neurais MLP, otimização de hiperparâmetros e
análise de regularização.

Na classificação, queremos prever a classe taxonômica do animal: aves,
mamíferos, répteis ou anfíbios. Na regressão, queremos prever a quantidade de
atropelamentos em uma unidade espacial e temporal.

## 0:50 - 1:40 | Dados e limpeza

A planilha original possuía 10.328 linhas. Durante a análise, identificamos
que os registros estavam duplicados em dois blocos. As cópias eram iguais,
exceto pelo campo técnico `OBJECTID`, criado pela camada GIS.

Por isso, não alteramos o arquivo original. Durante o carregamento, removemos
`OBJECTID` e eliminamos as duplicatas. A base final ficou com 5.164 ocorrências
únicas.

Depois filtramos somente os registros de fauna silvestre, obtendo 4.422
registros. As principais classes foram aves, répteis, mamíferos e anfíbios.

## 1:40 - 2:30 | Engenharia de atributos e pré-processamento

As variáveis numéricas foram padronizadas com StandardScaler e as variáveis
categóricas foram transformadas com One-Hot Encoding.

Também usamos o campo `data_dt` para criar novos atributos: mês numérico, dia
do ano, indicador de fim de semana, estação do ano e período chuvoso ou seco.

O campo `data_dt` não foi enviado diretamente para os modelos como texto. Ele
foi convertido em atributos numéricos e categóricos.

Os dados foram separados em aproximadamente 70 por cento para treinamento,
15 por cento para validação e 15 por cento para teste. O conjunto de teste foi
mantido separado durante as escolhas dos modelos.

## 2:30 - 4:00 | Classificação

Para a classificação, comparamos Regressão Logística, Random Forest e uma MLP.

A MLP inicial possui duas camadas ocultas, com 64 e 32 neurônios, ativação
ReLU e otimizador Adam. Depois do pré-processamento, ela recebe 34 entradas e
produz quatro saídas, uma para cada classe taxonômica.

As métricas usadas foram acurácia, precisão macro, recall macro, F1 Macro e
matriz de confusão. O F1 Macro foi importante porque as classes são
desbalanceadas e dá o mesmo peso para todas elas.

No teste, os resultados de F1 Macro foram aproximadamente:

- Regressão Logística: 0,228;
- Random Forest: 0,427;
- MLP: 0,350.

Neste experimento, o Random Forest obteve o melhor F1 Macro. A MLP teve
desempenho intermediário e a Regressão Logística apresentou o menor resultado.

Neste momento, mostrar a matriz de confusão e as curvas de loss e acurácia da
MLP.

## 4:00 - 5:20 | Regressão

Para a regressão, agregamos os registros por célula espacial de 500 metros e
mês. A variável alvo foi o número de atropelamentos em cada célula-mês.

Comparamos Regressão Linear, Random Forest Regressor e MLPRegressor.

No teste, a Regressão Linear obteve RMSE de aproximadamente 0,935 e R² de
0,046. A MLP obteve RMSE de 0,951 e R² de 0,012. O Random Forest Regressor
obteve RMSE de 1,040 e R² negativo.

Esses resultados indicam que, nesta primeira definição da unidade de análise,
os modelos explicam pouco da variação na quantidade de atropelamentos. Também
é importante lembrar que foram usadas apenas células-mês com ocorrências
observadas.

Neste momento, mostrar o gráfico de valores reais versus preditos e o gráfico
de resíduos.

## 5:20 - 6:50 | Otimização com Optuna

Na fase de otimização, usamos o Optuna para buscar uma configuração melhor para
a MLP de classificação.

Foram executados 50 trials. Foram testados o número de camadas, o número de
neurônios, a função de ativação, a taxa de aprendizado, a regularização e o
tamanho do batch.

A melhor configuração encontrada foi uma rede com arquitetura 32 e 64
neurônios, ativação ReLU, learning rate de aproximadamente 0,00195, alpha de
aproximadamente 0,000012 e batch size 32.

O melhor F1 Macro de validação foi 0,321 e o F1 Macro no teste foi 0,336.

O resultado mostra que a arquitetura inicial não é necessariamente a melhor,
mas também mostra que um ganho na validação não garante um ganho no teste.

## 6:50 - 8:10 | Regularização e overfitting

Comparamos uma MLP sem regularização com uma MLP usando regularização L2.

Sem regularização, o F1 Macro no teste foi aproximadamente 0,364. Com
regularização L2, usando alpha igual a 0,001, o F1 Macro subiu para 0,385 e a
acurácia subiu para aproximadamente 0,682.

Neste experimento, a regularização melhorou o desempenho no conjunto de teste.
As curvas de treino e validação foram usadas para observar a diferença entre o
aprendizado nos dados conhecidos e a generalização para dados não vistos.

Neste momento, mostrar as duas curvas de treinamento e explicar onde as curvas
se afastam.

## 8:10 - 9:20 | Discussão e limitações

Os modelos mais complexos não foram automaticamente superiores. O Random
Forest foi melhor na classificação inicial, enquanto a Regressão Linear teve o
melhor resultado na regressão.

As principais limitações são o desbalanceamento das classes, a ausência de
amostras confiáveis de locais sem atropelamento, a possível influência do
esforço amostral e a quantidade limitada de atributos ambientais e rodoviários.

Além disso, a divisão aleatória pode colocar registros geograficamente próximos
em conjuntos diferentes. Uma divisão espacial ou temporal seria uma análise
importante para avaliar melhor a generalização.

## 9:20 - 10:00 | Conclusão

Concluímos que foi possível construir um pipeline completo para classificação e
regressão, incluindo pré-processamento, engenharia de atributos, MLP, modelos
tradicionais, otimização e regularização.

Os resultados mostram que a qualidade dos atributos e da definição da unidade
de análise é tão importante quanto a arquitetura da rede neural.

Como próximos passos, podemos incorporar precipitação, temperatura, uso do
solo, volume de tráfego, características da rodovia e realizar uma validação
espacial ou temporal mais rigorosa.

Obrigado.
