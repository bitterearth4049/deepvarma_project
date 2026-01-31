# AGENTS

This repository contains research code for DeepVARMA variants. If you are contributing or automating tasks, follow these rules:

## Project Scope
- Focus on data ingestion, preprocessing, model training, and evaluation.
- Maintain compatibility with Hydra configuration in `conf/`.
- Keep output artifacts under `outputs/`.

## Style
- Prefer clear, minimal changes.
- Use ASCII text unless existing files require Unicode.
- Keep plots as PNGs via plotnine.

## Running
- Use `PYTHONPATH=src python -m deepvarma.pipeline.train` for training runs.
- Do not create virtual environments or lock package versions unless explicitly requested.

## References
- Theory source: https://arxiv.org/abs/2404.17615
