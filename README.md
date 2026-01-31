# DeepVARMA

DeepVARMA is a hybrid time series forecasting project that combines LSTM trend modeling with VARMA/VARMAX residual modeling. The implementation follows the theory described in: https://arxiv.org/abs/2404.17615

## Features
- DeepVARMA-re: LSTM trend + VARMA residuals
- DeepVARMA-en: LSTM exogenous encoding + VARMAX
- DeepVARMA (full): LSTM trend + LSTM exogenous encoding + VARMAX residuals
- data.gov.in ingestion for Agmarknet-style datasets
- Plotnine (grammar of graphics) PNG outputs

## Setup
Install dependencies:

```bash
pip install -r requirements.txt
```

## Run
The default configuration uses the Sheoraphuly potato price series.

```bash
PYTHONPATH=src python -m deepvarma.pipeline.train
```

Artifacts are written under `outputs/<date>/<time>/artifacts/`.

## Configuration
All configuration is handled via Hydra in `conf/`:

- `conf/config.yaml` selects dataset, model, training defaults
- `conf/dataset/*.yaml` contains dataset-specific settings
- `conf/model/*.yaml` contains DeepVARMA variants and hyperparameters
- `conf/training/default.yaml` contains training parameters

To switch datasets, update `conf/config.yaml` or override on the CLI:

```bash
PYTHONPATH=src python -m deepvarma.pipeline.train dataset=potato_haldwani_apmc
```

## Notes
- data.gov.in API access uses the provided public demo key in config.
- In-sample fit plots are not forecasts. For out-of-sample forecasts, use the test metrics and saved predictions.
