# RodoFauna ML

Projeto de aprendizado supervisionado para análise e predição de atropelamentos de fauna silvestre no Distrito Federal.

## Estrutura

- `data/raw/`: arquivos originais, preservados sem alterações.
- `data/processed/`: dados tratados e derivados.
- `notebooks/`: análises exploratórias e experimentos.
- `src/`: código reutilizável do pipeline.
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

## Status

Status: organização do projeto e preparação do ambiente.
