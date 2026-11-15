# Full experiment sweep on the Qwen3.5 representation (warm users, 2022-06-01 split).
# Each run writes outputs/runs/<rep>_<reducer><d>_<cutoff>.json
$env:PYTHONWARNINGS = "ignore"
Set-Location (Split-Path $PSScriptRoot -Parent)
$runs = @(
    @("none", 0),
    @("pca", 128), @("pca", 32), @("pca", 8), @("pca", 2),
    @("umap", 32), @("umap", 8), @("umap", 2), @("umap", 128)
)
foreach ($r in $runs) {
    python src/run.py --rep qwen35 --reduce $r[0] --dim $r[1] --baselines 2>&1 |
        Select-String "^run |userknn_only|hybrid_fixed|reranker  |top features"
}
