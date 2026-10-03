# Interpretable Intraday Signals for Equity-Index Direction Prediction

[简体中文](README_zh-CN.md) | English

This repository contains the analysis code associated with the research project
**“Interpretable Intraday Signals for Real-Time Open-to-Close Directional Prediction: Evidence from Chinese Equity Indices.”**

The project studies whether interpretable intraday price-pattern and breakout
signals observed by 13:30 can help classify the open-to-close direction of major
Chinese equity indices. It compares a deterministic rule benchmark with logistic
regression, chronological holdout evaluation, rolling validation, and external
index validation.

## Publication status

The associated manuscript has been accepted, but formal publication is still
pending. This repository is currently maintained as a private reproducibility
archive. No manuscript, acceptance letter, review material, or publication
agreement is included.

## Repository contents

- `intraday_index_direction_refactored.py` — final readability- and
  maintainability-focused version of the analysis pipeline.
- `intraday_index_direction_submission.py` — original analysis code preserved
  for provenance.
- `requirements.txt` — exact Python package versions used for the verified run.
- `data/README.md` — expected private-data layout, filename patterns, and
  required columns without distributing source data.
- `.gitignore` — prevents restricted data, generated outputs, manuscripts,
  credentials, and local caches from being committed.

The refactored version reorganizes repeated operations, centralizes constants,
uses explicit output paths, and provides a guarded `main()` entry point. It does
not change the research target, features, sample split, model settings,
prediction threshold, random seed, or reported results. The original and
refactored pipelines were verified to generate the same 43 output artifacts.

## Data availability

The minute-level source data were obtained through MyQuant and are not included
because redistribution permission has not been established. The analysis expects
three local data directories next to the Python script:

```text
1-沪深300数据/
2-上证指数数据/
3-深证成指/
```

Each directory contains annual CSV files for 2015–2025 following the filename
patterns used in the code. Users must obtain appropriately licensed data and
reproduce the required schema independently.

## Environment setup

Python 3.13 was used for the verified run. Install the recorded dependencies in
an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running the analysis

After placing appropriately licensed input files in the expected directories,
run the final version with:

```bash
python intraday_index_direction_refactored.py
```

Generated artifacts are written to `analysis_outputs_refactored/`. The original
submission version writes to `analysis_outputs/`.

## Reproducibility boundary

The repository preserves the analysis implementation and software environment,
but it cannot provide full third-party reproducibility without access to the
licensed source data. Results should not be interpreted as evidence of trading
profitability, investment performance, or causal effects.

## Author

Yingxuan Huang

## License

No open-source license is granted at this stage. Public distribution and reuse
remain subject to confirmation of publication rights and data-use restrictions.
