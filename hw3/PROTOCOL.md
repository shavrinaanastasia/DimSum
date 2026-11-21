# HW3 development protocol (sets and metrics fixed 21 Nov 2026)

> **Disclosure (added after review):**
> - One preliminary run of six variants on D0 test truth (`outputs/dev_D0_fast`, feature_fraction 0.8) happened before tuning, and `main` was defined afterwards. D0 is therefore not a clean held-out set for choosing between variants.
> - D1–D3 were first evaluated after every setting was fixed.
> - D3 replaced 2022-03-01 after only a baseline run at 2022-03-01.

All results under this protocol are **development results, not official benchmark scores**.

## Evaluation sets
The provided starter (`hw2/far-trans`, commit f6850861) defines every set. Common settings:
- warm users only (≥1 eligible buy before the cut-off);
- repeats allowed;
- 30-day window [cut-off, cut-off + 30 d);
- eligible assets = the starter's universe at the cut-off;
- truth = the starter's truth restricted to eligible assets.

| ID | Cut-off | Users | Role |
|---|---|---|---|
| T100 | 2021-06-01 | the starter's deterministic 100 | required teaching sample |
| D0 | 2021-06-01 | full warm cohort (`--max-users 0`) | primary development set |
| D1 | 2021-09-01 | full warm cohort | robustness |
| D2 | 2021-12-01 | full warm cohort | robustness |
| D3 | 2022-06-01 | full warm cohort | robustness |

D3 is 2022-06-01 rather than 2022-03-01 because investor embeddings exist for 2022-06-01. This was chosen before any comparison.

User selection never looks at future purchases or scores.

The `--fast` mode of `rerank3.py` builds rankings only for users with nonempty truth. This is purely a compute shortcut: empty-truth users score 0 for every method, so the all-user mean is identical. Reported runtimes come from full runs.

## Metrics
- **Primary:** mean nDCG@10 over all evaluated users, with empty truth counted as 0.
- **Secondary:** mean nDCG@10 over nonempty users, plus their count.
- Shown to 4 decimals. Paired bootstrap 95% CIs (10,000 resamples) for differences.

## Methods compared (same users, eligible assets, labels and rules)
- The starter's four baselines: Popularity, CBF, Latest buy, Latest buy or sell.
- The provided LLM comparison (starter LLM-Rec encoder vs original-text TF-IDF CBF, T100, reported by the starter) and my own matched controls:
  - asset Encoder (Qwen3.5) vs a TF-IDF text control on the same asset text;
  - investor Encoder (Qwen3.5, UMAP-32) vs the same reranker without it.
- My models, all one LightGBM LambdaRank on identical candidates:
  - `behaviour_hw2`: the HW2 feature set, trained on 12 monthly windows;
  - `rebuy`: plus re-buy features;
  - `rebuy_tfidf`, `rebuy_assetllm`, `rebuy_invllm`, `full`: plus the respective Encoder features.

## Training and leakage
- 12 monthly training windows before each cut-off, with label windows ending before the cut-off. The most recent window is used for early stopping.
- Then refit on all windows, averaged over 3 seeds.
- Hyper-parameters are chosen on D0's validation window only (inside training data), never on test truth.
- Ablation decisions use D0. D1–D3 are only reported, never used to choose.

## Settings fixed on validation (before any test number was read)
- Final method: `main` = base + rebuy + asset TF-IDF + asset LLM + investor LLM.
- LightGBM: lambdarank, lr 0.03, 31 leaves, min_data_in_leaf 50, **feature_fraction 0.5**, bagging 0.8/1, truncation 20, 3 refit seeds.
- 12 training windows (24 windows scored lower on validation: 0.6182 vs 0.6187, 3-seed averages).
- The v2 "extra" feature group was rejected on validation (0.6153 vs 0.6180).
- After the dev results came in, `main` was **not** replaced by a variant that happened to score higher on one dev set. The differences between variants are inside the bootstrap noise (see `hw3/outputs/dev_fast_*`).
