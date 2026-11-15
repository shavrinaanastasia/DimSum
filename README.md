# DimSum — How Much Investor Representation Is Enough?

**Dimensionality Reduction for LLM-Based Financial Recommendation**

Team project for the LLM4Rec course (HSE University, 2026), built on the public [FAR-Trans](https://researchdata.gla.ac.uk/1658/) dataset ([paper](https://arxiv.org/abs/2407.08692)).

## Team

| Member | Individual research question |
|---|---|
| Анастасия Шаврина (Anastasia Shavrina) | How does compressing LLM-derived investor representations to 128, 32, 8 or 2 dimensions affect Top-10 recommendation quality (nDCG@10) compared to the uncompressed representation? |
| Анна Шрейбер | Does the effect of compression differ between investors with sparse and extensive transaction histories? |

## Pipeline (Session 2)

Updated after instructor feedback: the score is judged against the classical baseline, UMAP replaces PCA as the main reducer, and Qwen3.5 replaces Qwen3. Cold-start users are Anna's part, so this pipeline is evaluated on warm users (≥1 buy before the cut-off).

1. **Split.** Model input uses rows dated before the cut-off only. The eligible universe is the assets with a price and an attribute record before the cut-off. Truth is the eligible assets bought in the window. Primary split: 2022-06-01 → 2022-12-01. Training splits use the same window length at earlier cut-offs (2021-12-01 for validation; 2021-09-01, 2021-06-01 and 2021-03-01 for training), so every training window ends on or before the test cut-off.
2. **Prompted encoder (LLM).** Each investor's profile and purchase history are serialised as text and encoded by frozen **Qwen3.5-0.8B** with a zero-shot prompt (last-token embedding). There is no fine-tuning. See `src/embed.py`.
3. **Reduction.** **UMAP** to *d* dimensions (main). **PCA** to *d* ∈ {128, 32, 8, 2} is kept as a side experiment. Both are fit on pre-cut-off users only.
4. **Candidates.** Each user gets the assets they bought before, plus the top assets of their 50 nearest neighbours in the reduced space, item-kNN suggestions and recently popular assets.
5. **Learned reranker.** A LightGBM LambdaRank model trained on the earlier splits. Its features are repeat-purchase signals, popularity and trend, price returns, attribute (CBF) similarity, item-kNN/EASE scores and the **embedding-neighbour score and rank**. It outputs the Top-10 per user.

## How to run

```bash
pip install numpy pandas scikit-learn umap-learn lightgbm transformers torch
# classical baselines + reranker on the purchase-vector representation
python src/run.py --data data/FAR-Trans --baselines
# LLM path: export texts (CPU) -> encode (GPU, notebooks/encode_qwen35_colab.ipynb) -> assemble
python src/embed.py export
python src/embed.py assemble --tag qwen35
python src/run.py --rep qwen35 --reduce umap --dim 32
python src/run.py --rep qwen35 --reduce pca  --dim 32
# best pipeline, and the FAR-Trans paper protocol on five benchmark dates
python src/run.py --rep qwen35 --reduce umap --dim 8 --tag v3
powershell -File scripts/paper_protocol.ps1
```

Each run writes `outputs/runs/<name>.json` (per-user nDCG@10, 95% bootstrap CI, log) and `<name>_top10.csv`.

## Best pipeline (v3)

Upgrades over the first version:
- **Larger candidate pool.** Top-50 per source, with iALS and EASE added as sources. Candidate recall rose from 0.91 to 0.97.
- **Tuned iALS.** Implicit ALS (k = 64, reg = 100, tuned on non-test dates with `scripts/tune_ials.py`), used as a score and rank feature.
- **Investor risk-fit features.** Risk level, capacity, customer type, the average risk level of an asset's past buyers, and popularity within the user's risk segment.

**Our protocol** (2022-06-01 → 2022-12-01, 2,588 warm users, repeat purchases count):

| Pipeline | nDCG@10 | vs CBF (0.456) | vs first reranker (paired 95% CI) |
|---|---|---|---|
| Reranker, behaviour only (v3) | 0.6293 | +0.174 | +0.006 [+0.002, +0.010] |
| Reranker + Qwen3.5 → PCA-8 (v3) | 0.6301 | +0.175 | +0.007 [+0.003, +0.011] |
| **Reranker + Qwen3.5 → UMAP-8 (v3)** | **0.6327** | **+0.177** | **+0.009 [+0.005, +0.013]** |

With v3, UMAP-8 Qwen3.5 embeddings beat the behaviour-only reranker by +0.0034 (paired 95% CI [+0.0001, +0.0068]). A random-vector control (same pipeline, random 64-d vectors per investor text, UMAP-8) scores 0.6313. Qwen3.5 beats this control by only +0.0014 (95% CI [−0.0010, +0.0037]), so the LLM's own contribution is not significant. Most of the small gain comes from the reranker having an extra neighbour-based input, not from what the LLM understands about the investor.

**FAR-Trans paper protocol** (assets the user never bought; already-bought assets never recommended; 6-month window). This runs on five of the paper's 61 dates (`scripts/paper_protocol.ps1`, `python src/paper_compare.py v2`). The reference is the public benchmark of 50+ methods ([TechyNilesh/far-trans-benchmark](https://github.com/TechyNilesh/far-trans-benchmark), ACML 2026 workshop), whose best single method overall is MF-BPR (0.354 mean over 61 dates). Our evaluated-user counts match theirs (for example 1,692 vs 1,692).

| Date | MF-BPR | Best of 50+ public methods on that date | Ours v2 | Ours v3 |
|---|---|---|---|---|
| 2019-11-21 | 0.428 | 0.428 | 0.407 | 0.385 |
| 2020-05-22 | 0.411 | 0.414 | 0.429 | 0.422 |
| 2020-11-23 | 0.386 | 0.388 | 0.398 | 0.401 |
| 2021-05-25 | 0.221 | 0.223 | 0.246 | 0.251 |
| 2021-11-23 | 0.298 | 0.313 | 0.301 | 0.307 |
| **Mean** | **0.349** | 0.353 | **0.356** | 0.353 |

Our reranker beats MF-BPR on 4 of 5 dates. It loses on 2019-11-21, where its training splits have only about 400 warm users each.

## First results (2022-06-01 → 2022-12-01, 2,588 warm users, nDCG@10)

**Baselines:** random 0.006; popularity (90 days) 0.362; **CBF attributes 0.456**; repeat 0.571; user-kNN on purchase vectors 0.574; hybrid with fixed weights 0.608.

**Learned reranker** (3 seeds). "Δ vs behaviour" is the paired bootstrap 95% CI against the reranker that uses purchase vectors only (`python src/compare.py`).

| Representation | Reducer | d | user-kNN only | Reranker | Δ vs behaviour |
|---|---|---|---|---|---|
| Purchase vectors (behaviour) | none | — | 0.574 | 0.6234 | — |
| Purchase vectors | PCA | 128 / 32 / 8 / 2 | | 0.6268 / 0.6226 / 0.6243 / 0.6235 | +0.003 / −0.001 / +0.001 / +0.000 |
| Qwen3.5-0.8B | none | 1024 | 0.482 | 0.6263 | +0.003 [+0.000, +0.006] |
| Qwen3.5-0.8B | **UMAP** | 128 / 32 / 8 / 2 | 0.469 / 0.466 / 0.467 / 0.467 | 0.6253 / 0.6222 / 0.6262 / 0.6260 | +0.002 / −0.001 / +0.003 / +0.003 |
| Qwen3.5-0.8B | PCA (side) | 128 / 32 / 8 / 2 | 0.478 / 0.470 / 0.448 / 0.319 | 0.6252 / 0.6262 / **0.6285** / 0.6266 | +0.002 / +0.003 / **+0.005 [+0.002, +0.008]** / +0.003 |
| Random vectors (control) | UMAP | 8 | | 0.6260 | +0.003 [−0.001, +0.006] |

**Findings**

- The best pipeline is the reranker with Qwen3.5 → PCA-8: **0.6285**. That is +0.173 over CBF (+38%) and +0.267 over popularity.
- Most of the gain over the baselines comes from the reranker and its repeat, popularity and EASE features. Every variant scores between 0.622 and 0.629.
- On its own, the LLM neighbour signal is weaker than the purchase-vector neighbour signal (0.48 vs 0.57). Inside the reranker it adds at most about +0.005.
- The random-vector control scores 0.626, so the LLM's margin over random vectors (≤ +0.0025) is not significant.
- Compression is almost free inside the reranker. UMAP holds kNN quality down to d = 2 (0.467), while PCA kNN collapses at d = 2 (0.319); the reranker recovers the loss either way.

## Homework 2 (starter protocol)

`hw2/far-trans` and `hw2/far-trans-llm` are the instructor's starter, downloaded unmodified from `dryjins/RecSys-LLMs` at commit f6850861. The encoder `hw2/far-trans-llm/embedding_encoder.py` ranks assets with Qwen3.5 embeddings → UMAP or PCA (d = 128/32/8/2) → user-kNN, written beside the four provided baselines.

```bash
cd hw2/far-trans && python starter.py run --data-dir ../../data/FAR-Trans --output-dir ../outputs/baseline
cd ../far-trans-llm && python embedding_encoder.py --data-dir ../../data/FAR-Trans --output-dir ../outputs/encoder100 \
    --emb ../../outputs/emb/qwen35_2021-06-01.npy --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17
cd ../.. && python hw2/summarize.py hw2/outputs/encoder100 hw2/outputs/baseline && python hw2/check_one_user.py hw2/outputs/encoder100 llm_uknn_umap32
```

Main method, `hw2/far-trans-llm/reranker_encoder.py`: the compressed-embedding neighbour score becomes a feature of the team's LightGBM LambdaRank reranker. The reranker is trained on the 2021-03/04/05 windows, and a no-LLM control is included.

```bash
cd hw2/far-trans-llm && PYTHONHASHSEED=0 python reranker_encoder.py --data-dir ../../data/FAR-Trans \
    --output-dir ../outputs/reranker100 --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17
cd ../.. && python hw2/vs_baselines.py hw2/outputs/reranker100 reranker_llm_umap32 reranker_behaviour
```

Results, nDCG@10 (all users / nonempty truth):

| Method | 100 users | Full warm cohort (10,404) |
|---|---|---|
| Best provided baseline | Popularity 0.0286 / 0.2599 | Latest buy 0.0526 / 0.4648 |
| Reranker + LLM-UMAP-32 | 0.0386 / 0.3506 | **0.0543 / 0.4798** |

On the full cohort, the UMAP-32 reranker beats Latest buy by +0.0149 on nonempty users, 95% CI [+0.0045, +0.0254]. All nine LLM variants beat all four baselines. The reranker without the LLM feature scores the same (0.4782), though, so the gain comes from the reranker, not the LLM.

Add `--max-users 100000` (and a fresh output directory) to run the full warm cohort. `python hw2/paired.py <dir> [reference]` runs the paired non-inferiority analysis, with reference `llm_uknn_full` or `reranker_llm_full`. The report is `hw2/Shavrina_hw2_report.pdf`.

## Status and plan

| Session | Deliverable | Status |
|---|---|---|
| 1 | Research question, data example, study design (individual reports) | Done |
| 2 | Loader, temporal split, evaluator, baselines, learned reranker, Qwen3.5 encoder, UMAP/PCA sweep | Done |
| 3 | UMAP vs PCA at all *d*, ablations, failure analysis | Planned |
| 4 | Final Top-10 predictions, report, presentation | Planned |

## Data

The FAR-Trans dataset (CC-BY 4.0) is **not** stored in this repository. Download `FAR-Trans.zip` from <https://researchdata.gla.ac.uk/1658/> (DOI [10.5525/gla.researchdata.1658](https://doi.org/10.5525/gla.researchdata.1658)) and extract it locally.

## Citation

J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An investment dataset for financial asset recommendation," IJCAI-2024 Workshop on Recommender Systems in Finance (Fin-RecSys), 2024. https://arxiv.org/abs/2407.08692
