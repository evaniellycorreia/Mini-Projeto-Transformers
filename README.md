# Atenção em séries temporais: Transformer no Jena Climate

Previsão multivariada do conjunto Jena Climate (2009–2016) com um codificador
Transformer em PyTorch: próximo token (10 min) e próximas 24 h.

## Como reproduzir

~~~bash
git clone (https://github.com/evaniellycorreia/Mini-Projeto-Transformers)
cd <nome-do-repositorio>
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python 01_analise_dados.py --seeds 1 2 3
~~~

O dataset é baixado automaticamente para `data/` na primeira execução
(fonte: https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip).
As figuras são salvas em `outputs/seed_N/`.

## Opções

| Argumento | Padrão | Descrição |
|---|---|---|
| `--seeds` | 42 | Uma ou mais sementes (ex.: `--seeds 1 2 3`) |
| `--epochs` | 5 | Épocas dos modelos principais |
| `--window-step` | 24 | Usa 1 a cada N janelas no treino |
| `--skip-sweep` | — | Pula a varredura de hiperparâmetros |

## Resultados (MSE padronizado no teste, média de 3 sementes)

| Objetivo | Transformer | Baseline (último valor) |
|---|---|---|
| Próximo token | 0,0479 ± 0,0003 | 0,0606 |
| 24 h | 0,2825 ± 0,0019 | 0,6230 |

## Estrutura

- `01_analise_dados.py`: dados, modelo, treino, avaliação e figuras
- `relatorio/`: relatório em LaTeX, figuras e PDF

## Ambiente

Python <versão>, executado em CPU. Valores podem variar ligeiramente entre
máquinas e versões do PyTorch.
