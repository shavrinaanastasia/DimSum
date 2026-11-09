# DimSum — How Much Investor Representation Is Enough?

**Dimensionality Reduction for LLM-Based Financial Recommendation**

Team project for the LLM4Rec course (HSE University, 2026), built on the public [FAR-Trans](https://researchdata.gla.ac.uk/1658/) dataset ([paper](https://arxiv.org/abs/2407.08692)).

## Team

| Member | Individual research question |
|---|---|
| Анастасия Шаврина (Anastasia Shavrina) | How does compressing LLM-derived investor representations to 128, 32, 8 or 2 dimensions affect Top-10 recommendation quality (nDCG@10) compared to the uncompressed representation? |
| Анна Шрейбер | Does the effect of compression differ between investors with sparse and extensive transaction histories? |

## Shared setup

- **Representation:** each investor's profile and pre-cut-off transaction history are serialised as text and encoded with a frozen LLM embedding model (Qwen3-Embedding-0.6B, 1024-d).
- **Change under test:** PCA compression of the embedding to *d* ∈ {128, 32, 8, 2}; the uncompressed 1024-d embedding is the control.
- **Ranker:** user-kNN in embedding space, identical for every variant.
- **Output and metric:** the common Top-10 list per user, scored by nDCG@10 against assets bought in the future evaluation window.
- **Baselines:** popularity and random; a TF-IDF control on the same text separates "LLM encoding" from "dimension".
- **Ablations:** Gaussian random projection, Matryoshka truncation (128 and 32 only), prompt templates, names-included contamination check.

The official split and eligibility protocol will be fixed by the instructor before Session 2; until then all counts use an assumed cut-off (2022-06-01) and are not official.

## Status and plan

| Session | Deliverable | Status |
|---|---|---|
| 1 | Research question, data example, study design (individual reports) | Done |
| 2 | Data loader, temporal split, evaluator, popularity/random baselines, 1024-d kNN control | Planned |
| 3 | PCA at all four *d*, ablations, non-LLM control, subgroup analysis, failure analysis | Planned |
| 4 | Final Top-10 predictions, report, presentation | Planned |

## Data

The FAR-Trans dataset (CC-BY 4.0) is **not** stored in this repository. Download `FAR-Trans.zip` from <https://researchdata.gla.ac.uk/1658/> (DOI [10.5525/gla.researchdata.1658](https://doi.org/10.5525/gla.researchdata.1658)) and extract it locally.

## Citation

J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An investment dataset for financial asset recommendation," IJCAI-2024 Workshop on Recommender Systems in Finance (Fin-RecSys), 2024. https://arxiv.org/abs/2407.08692
