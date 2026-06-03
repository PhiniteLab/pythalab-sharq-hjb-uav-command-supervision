# Experiment Package

This directory is the single place for claim statements and result files.

Canonical experiment paths are rooted here. No alternate result root is part of this package.

```text
experiments/
  claims/                 # plain-language claims and their code/result links
  results/
    raw/                  # backend runner outputs
    data/                 # compact CSV/JSON/JSONL result data
    tables/               # CSV table outputs
    figures/              # PNG figure outputs only
```

## What belongs here

- Claims that describe what the code demonstrates.
- Raw benchmark results from the backend experiment runner.
- Compact result data used by table and figure scripts.
- CSV tables.
- PNG figures.

## What does not belong here

- Private notes.
- Upload bundles.
- Local virtual environments.
- Dependency folders.
- Non-PNG figure formats.

## Main commands

```bash
python scripts/sync_results.py
python scripts/export_tables.py
python scripts/write_figures.py
sha256sum -c reproducibility/MANIFEST.sha256
```
