# HW3: Test Your Project Idea (Anastasia Shavrina)

All results are **development results** under the starter's teaching protocol, not official benchmark scores.
The protocol (`PROTOCOL.md`) was locked before any method comparison.

## Requirements

Run every command from the repository root.

- **Data:** FAR-Trans in `data/FAR-Trans/`. It is not in the repository; get it from the dataset DOI.
- **Starter:** in `hw2/far-trans`, with `hw2/far-trans-llm` alongside it.
- **Investor embeddings:** the HW2 Qwen3.5 investor embeddings in `outputs/emb/`, made with `notebooks/encode_qwen35_colab.ipynb`.
- **Environment variables:** `PYTHONHASHSEED=0`, plus `HF_HUB_OFFLINE=1` once the model is cached.

## Steps

| Step | Command | Output |
|---|---|---|
| Asset Encoder (Qwen3.5-0.8B, about 20 min on CPU) | `python hw3/embed_assets.py` | `hw3/cache/asset_emb.pkl` |
| Validation-only tuning (D0 window 2021-05) | `python hw3/tune.py`, `python hw3/tune2.py 12`, `python hw3/tune_extra.py` | `outputs/tune_2021-06-01.csv` |
| T100 (starter's 100 users), all variants | `python hw3/rerank3.py --cutoff 2021-06-01 --max-users 100 --output-dir hw3/outputs/T100_full` | `outputs/T100_full/` |
| Full cohort D0, with runtime | `python hw3/rerank3.py --cutoff 2021-06-01 --configs main,rebuy_tfidf --output-dir hw3/outputs/D0_full` | `outputs/D0_full/` |
| Dev sets D0–D3, all variants (compute shortcut) | `python hw3/rerank3.py --cutoff <c> --fast --output-dir hw3/outputs/dev_fast_<c>`, for c in 2021-06-01, 2021-09-01, 2021-12-01, 2022-06-01 | `outputs/dev_fast_*/per_user.csv` |
| Paired bootstrap CIs | `python hw3/compare.py hw3/outputs/dev_fast_<c> [reference]`, `python hw3/pooled.py` | stdout |
| Failure analysis | `python hw3/failure.py` | `outputs/failure_by_group.csv` |
| Candidate and embedding coverage | `python hw3/coverage.py` | `outputs/coverage.csv` |
| Hand check of one user | `python hw3/check_one_user.py hw3/outputs/T100_full main` | stdout |
| Starter baselines, stand-alone (protocol hashes) | starter `run.py --max-users 0 --cutoff <c>` | `outputs/baseline_full_*/protocol.json` |

`headroom.py` and `heuristics.py` are planning analyses only, not reported results.

## Per-user score files

The per-user scores for the dev sets, D0_full and the baseline runs (all files over 1 MB) are packed in `hw3/outputs/per_user_results.zip`. Unzip it into `hw3/outputs/` before running `compare.py`, `pooled.py` or `failure.py`.

## Not in the repository (size)

- `hw3/cache/` (cached feature frames, about 1.6 GB): rebuilt automatically.
- The large `predictions.csv` and `contexts.jsonl` files: re-created by the commands above.
