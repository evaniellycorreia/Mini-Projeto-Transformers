# Atenção em séries temporais: Transformer no Jena Climate

Previsão multivariada do conjunto Jena Climate (2009–2016) com um codificador
Transformer em PyTorch, com dois objetivos: prever o próximo token (10 min) e
prever as próximas 24 h. Os pesos de atenção são extraídos e analisados em
mapas de calor.

Repositório: https://github.com/evaniellycorreia/Mini-Projeto-Transformers

Relatório: [`Relatorio/2026-relatorio-EVANIELLYCORREIA.pdf`](Relatorio/2026-relatorio-EVANIELLYCORREIA.pdf)

## Como reproduzir

~~~bash
git clone https://github.com/evaniellycorreia/Mini-Projeto-Transformers.git
cd Mini-Projeto-Transformers
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python 01_analise_dados.py --seeds 1 2 3
~~~

O dataset é baixado automaticamente para `data/` na primeira execução
(fonte: https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip).
As figuras são salvas em `outputs/seed_N/`.

Teste rápido (poucos minutos):

~~~bash
python 01_analise_dados.py --seeds 1 --skip-sweep --epochs 1
~~~

## Opções

| Argumento | Padrão | Descrição |
|---|---|---|
| `--seeds` | 42 | Uma ou mais sementes (ex.: `--seeds 1 2 3`) |
| `--epochs` | 5 | Épocas dos modelos principais |
| `--window-step` | 24 | Usa 1 a cada N janelas no treino |
| `--skip-sweep` | — | Pula a varredura de hiperparâmetros |
| `--data-dir` | `data/` | Pasta do dataset |
| `--output-dir` | `outputs/` | Pasta das figuras |

## Metodologia (resumo)

- **Dados:** 420 551 registros, 14 variáveis, a cada 10 min. Os valores `-9999` de vento
  foram interpolados, e `wv` foi limitado por `max. wv`.
- **Divisão:** cronológica, com 80% para treino e 20% para teste. A padronização é ajustada só no treino.
- **Janelas:** entrada de 144 passos (24 h). O alvo é de 1 passo (próximo token) ou de 144 passos (24 h).
- **Modelo:** projeção linear, codificação posicional senoidal, 2 blocos codificadores
  (4 cabeças, `n = 64`) e uma camada linear de saída aplicada ao último token.
- **Perda:** MSE nos dados padronizados. O otimizador é o Adam (1e-3), com corte de gradiente em 1,0.
- **Baseline:** repete o último valor observado.

## Resultados (MSE padronizado no teste, média de 3 sementes)

| Objetivo | Transformer | Baseline (último valor) |
|---|---|---|
| Próximo token | 0,0479 ± 0,0003 | 0,0606 |
| 24 h | 0,2825 ± 0,0019 | 0,6230 |

O Transformer reduz o erro em cerca de 21% no próximo token e em 55% em 24 h.

## Estrutura

- `01_analise_dados.py`: dados, modelo, treino, avaliação e figuras
- `requirements.txt`: dependências com versões
- `Relatorio/`: relatório em LaTeX, figuras (semente 1) e PDF (`2026-relatorio-EVANIELLYCORREIA.pdf`)

## Ambiente

Python 3.x, executado em CPU. Os valores podem variar ligeiramente entre máquinas e
versões do PyTorch.
