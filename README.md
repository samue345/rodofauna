# RodoFauna ML

Projeto de aprendizado supervisionado para análise e predição de atropelamentos de fauna silvestre no Distrito Federal.

## Estrutura

- `data/raw/`: arquivos originais, preservados sem alterações.
- `data/processed/`: dados tratados e derivados.
- `notebooks/`: análises exploratórias e experimentos.
- `src/`: código reutilizável do pipeline.
  - `src/models/classification/`: classificação tradicional e MLP.
  - `src/models/regression/`: regressão com modelos tradicionais e MLP.
  - `src/models/optimization/`: otimização de hiperparâmetros com Optuna.
  - `src/models/regularization/`: comparação de MLPs com e sem regularização.
- `models/`: modelos treinados e métricas.
- `figures/`: gráficos e mapas.
- `reports/`: relatório técnico e materiais finais.

## Fonte dos dados

Os dados são provenientes do Programa RodoFauna do IBRAM/DF. O enunciado do trabalho e a planilha original estão em `data/raw/`.

## Objetivos

1. Classificar o grupo taxonômico da fauna silvestre atropelada.
2. Estimar a quantidade de atropelamentos por unidade espacial e temporal.
3. Investigar áreas com maior concentração de ocorrências.
4. Comparar MLPs com modelos tradicionais de Machine Learning.

## Como começar

```bash
pip install -r requirements.txt
```

Os experimentos serão executados pelos notebooks e scripts em `notebooks/` e `src/`.

## Execução com Docker

Com Docker Desktop instalado, execute na raiz do projeto:

```bash
docker compose up --build
```

O container executa `python -m src.main`. Como o projeto é montado no
container, os resultados continuam disponíveis localmente em
`reports/generated/`.

Para executar em segundo plano:

```bash
docker compose up --build -d
```

Para remover o container:

```bash
docker compose down
```

## Status

Fases 3, 4, 5 e 6 concluídas na primeira versão. A otimização de hiperparâmetros
testa agora de uma a três camadas ocultas e diferentes quantidades de neurônios:
classificação multiclasse e regressão com MLP são comparadas a modelos
tradicionais, e o Optuna otimiza a MLP de classificação usando apenas treino e
validação. A regressão agrega os registros por célula espacial de 500 m e mês,
usando a quantidade de atropelamentos como variável alvo. A fase 6 compara uma
MLP sem regularização com uma MLP usando penalização L2. Durante a preparação,
o campo técnico `OBJECTID` é removido e as duplicatas da exportação GIS são
eliminadas; o arquivo original em `data/raw/` é preservado.
