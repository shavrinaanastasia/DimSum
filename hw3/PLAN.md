# HW3 plan: Test Your Project Idea (draft, 21 Nov 2026)

## What is required (homework3.md + Session 3 slides + professor's message)

- **Main experiment** on our existing question: the LLM is an **Encoder**; we may add a Ranker or both.
  - We must say which role the LLM plays and what it changes.
  - The model must actually run inference.
- **One focused ablation:** change or remove one component.
  - If we use both an Encoder and a Ranker, we must separate their contributions.
- **Failure analysis:** cases where the method fails or changes nothing.
  - Record invalid outputs, fallbacks and coverage.
- **Development evaluation:**
  - Start from the HW2 setup: 100 users, cut-off 2021-06-01, 30 days, repeats allowed.
  - We may use a larger development sample or extra windows, but **we must fix them before comparing methods** and apply them to every method.
  - Label these results "development", not official.
- **Metrics:** report the four baselines, the provided LLM comparison and a matched control.
  - Keep users, eligible assets, labels and rules identical across methods.
  - nDCG@10 over all users (empty = 0) is primary.
  - Also report nonempty nDCG@10 with the user count, to 4 decimals.
  - Save settings and protocol hashes.
- **Personal verification** of one result with a hand check.
- **Reproducibility record:** model version, settings, hardware.
  - Report training/preparation time separately from recommendation time.
- **Report:** PDF of at most 5 pages plus the session export.
  - Sections: Title, Abstract, Introduction, Related Work, Method, Experiments, Discussion, AI Usage Disclosure, References.
  - Include name, team, repo, experiment commit, team decisions, open questions and my contribution.
- **Slides:** the claim must follow the form "for [users], [change] moved [metric] relative to [control] under [conditions]".
  - An honest negative result is valid.
  - "Research quality is not the same as a high leaderboard score."
- **Final benchmark (project grade):** 50 points, rank-based on the **official unrounded nDCG@10**.
  - Ties are broken by shorter recommendation-generation runtime.
  - The official set comes ≥100 h before the deadline. Until then, tune only on development data.

## Where the score can come from (hw3/headroom.py, hw3/heuristics.py; planning only)

Approximate protocol for these planning numbers: all users who bought before the cut-off, all assets eligible.

| cut-off | nonempty share | repeat share of future buys | Latest buy (nonempty) | repeat oracle (nonempty) |
|---|---|---|---|---|
| 2021-03-01 | 10.6% | 56% | 0.361 | 0.493 |
| 2021-06-01 | 4.8% | 63% | 0.448 | 0.657 |
| 2021-09-01 | 5.2% | 67% | 0.458 | 0.661 |
| 2021-12-01 | 3.6% | 73% | 0.494 | 0.722 |
| 2022-03-01 | 7.1% | 78% | 0.580 | 0.770 |
| 2022-06-01 | 4.5% | 83% | 0.607 | 0.825 |

- Empty users score 0 for every method, so **only the ~5–10% nonempty users decide the ranking**.
- Most future buys are re-buys, and the repeat share grows over time.
  - A perfect ordering of the user's *own* past assets would add about +0.2 nonempty nDCG over Latest buy.
  - This is the biggest lever.
- Hand-made orderings barely move it: 60-day exponential decay gives +0.003, and periodicity rules hurt.
  - Our learned LambdaRank reranker gave +0.015 (2021-06-01, full cohort).
  - The gain must come from a **better learned model**, not rules.
- The remaining 17–44% of future buys are new assets. They fill the free slots for users with fewer than 10 past assets.

## Proposed development protocol (fix this before any comparison)

1. **Required:** the starter's 100-user sample at 2021-06-01, 30 days.
2. **Primary development set:** the starter's full warm cohort at 2021-06-01 (10,404 users, 1,178 nonempty), as in HW2.
3. **Robustness windows:** the same starter rules at cut-offs **2021-09-01, 2021-12-01 and 2022-03-01**.
   - These show whether a gain holds outside one month, since the official cut-off is unknown.
   - Training data always ends before each cut-off.

## Ideas to test (ranked by expected score gain)

A. **More training windows for the reranker.** Today it trains on 3 monthly windows; this would use about 12–24 rolling windows.
   - Recency weighting; fixed seeds and averaged refits.
   - Expected: the largest and safest gain.

B. **Repeat-specific features:**
   - per (user, asset) inter-buy gaps, regularity, and buys in the same calendar month last year;
   - the asset's share of the user's buys in the last 1/3/6 months;
   - net position (bought minus sold units);
   - whether a sell came after the last buy;
   - user-level re-buy rate.

C. **New-asset slot filling:** a separate candidate scorer for not-yet-bought assets, used only in the free slots.
   - Item-kNN, EASE, segment popularity and the LLM asset similarity in (D).

D. **LLM as an asset-side Encoder (new).** Embed each asset's supplied attributes with Qwen3.5-0.8B (PromptEOL), giving asset–asset semantic similarity to the user's portfolio as a feature.
   - It mainly helps new-asset candidates. Matched control: the same feature built from TF-IDF of the same text.

E. **Keep the HW2 investor Encoder** (UMAP-32 user embedding kNN) as a feature: a hybrid with a clean ablation.
   - In HW2 it added under 1%, an honest near-zero result.

F. **(Optional) LLM Ranker on the top-k** (LLMRank-style, Qwen3.5-0.8B) only for users with many tied candidates.
   - Risk: slow on CPU, invalid outputs, and the provided Ranker did not beat CBF.
   - Runtime counts in the tie-break, so I'd treat this only as a measured negative control.

G. **(Optional) News RAG:** the slides call it optional. FAR-Trans has no issuer news, so the expected gain is low; skip unless time allows.

## Team split: what Anna does

What Anna did before (HW1 and HW2 team tables):
- She built the shared profile rule, the cold-start split (warm / cold-profile / cold-no-information) and the diversity subgroups.
- Her research question: does the effect of compression (and of the LLM) differ between investors with **sparse** and **extensive** histories?
- In HW2 we split users: I took warm investors, and she took cold-start and sparse-history users.

Sizes of her groups. These are planning numbers: future buyers, i.e. users with at least one buy in the 30 days, by how many distinct assets they bought before the cut-off.

| cut-off | future buyers | cold (0) | 1 | 2–4 | 5–9 | 10+ |
|---|---|---|---|---|---|---|
| 2021-06-01 | 1,376 | 167 | 102 | 261 | 326 | 520 |
| 2021-09-01 | 1,647 | 280 | 126 | 283 | 374 | 584 |
| 2022-03-01 | 2,133 | 112 | 201 | 492 | 587 | 741 |

Share of *new* (never-bought) assets in their purchases: 54–67% for 1-asset users, 34–45% for 2–4, and 14–32% for 10+.
- Sparse users therefore depend on new-asset recommendations, not on reordering their history.
- Cold users are 5–17% of buyers. The starter's teaching cohort is warm-only, but the official set may include them.

**Anna's tasks**

1. **New-asset slot filler (idea C).** Score assets the user has *not* bought; they fill the slots left after the user's own past assets.
   - Signals: item-kNN/EASE from the user's assets, risk-segment popularity, recent popularity trend, and the LLM asset-similarity feature I provide (idea D).
   - Output: one score per (user, unseen asset), which I merge as a reranker feature/candidate source.
   - Matched control: the same filler without the LLM feature.
2. **Cold-start path.** A ranking for users with no pre-cut-off buy:
   - *cold-profile*: questionnaire-only, using a profile-aware popularity (risk level, capacity, customer type) plus her profile-text LLM embedding if available;
   - *cold-no-information*: plain recent popularity.
   - Evaluate it on dev cold users at the same cut-offs. Ready in case the official set includes cold users.
3. **Sparse-vs-extensive subgroup analysis (her research question)** on the shared dev protocol.
   - Report nDCG@10 per history bin (1 / 2–4 / 5–9 / 10+ distinct assets) for the four baselines, the full model, the no-LLM ablation and the compression variants.
   - Question: does the LLM help sparse users more?
   - This doubles as our **failure analysis** by user type.
4. **Personal verification** of one of her own results (hand-worked nDCG for one sparse user), and her own report sections.

**What I do (Anastasia):** ideas A, B, D, E, and the integration:
- the multi-window reranker and the re-buy features;
- the asset-side LLM encoder, of which she gets a copy of the embeddings;
- the investor encoder ablation;
- the locked dev protocol and the evaluation scripts. She uses the same scripts, so the numbers stay comparable.

**Grading is individual.** Anna is graded separately, so my HW3 report must be complete without her deliverables:
- my own main experiment, ablation, failure analysis (including a by-history-size breakdown of my own model) and personal verification;
- new-asset slots are handled inside my reranker by its own candidate sources;
- Anna's filler or cold-start scores are an optional extra that I add only if they arrive in time and help on dev.

**Shared and decided together:**
- the dev protocol (above), fixed before comparisons;
- one final pipeline submitted for the benchmark;
- the ablation definition;
- the interface: Anna's filler and cold-start scores enter as columns `anna_new_score` and `anna_cold_rank` per (user, asset).

## Ablation and failure analysis

- **Ablation (one component):** remove the LLM features (asset Encoder D and investor Encoder E) and keep everything else fixed.
  - That is the behaviour-only control.
  - A second view: D alone versus E alone.
- **Failure analysis:**
  - nonempty users where the model ranks below Latest buy;
  - users whose purchases are all new assets;
  - users with a single past asset;
  - per-window variation;
  - coverage of truth items by candidates.

## Status (21 Nov 2026)
- **Done:**
  - The protocol was locked and the settings were fixed on validation (see PROTOCOL.md).
  - D0–D3 fast runs: `hw3/outputs/dev_fast_*`, compared with `hw3/compare.py`.
  - Failure analysis: `hw3/failure.py`, written to `hw3/outputs/failure_by_group.csv`.
- **Result:**
  - `main` beats Latest buy on all four dev sets (nonempty difference +0.017 to +0.050), and every CI excludes 0.
  - The LLM features (asset and investor Encoders) do not measurably change nDCG@10 against the TF-IDF and no-LLM controls: all differences are within ±0.004, and every CI includes 0.
  - The gain is largest for users with short histories and for users whose next buys are all new assets. It is about zero for 10+-asset users at D0.
- **Remaining:**
  - T100 full run and full-cohort runtime;
  - the hand check;
  - the PDF report.
