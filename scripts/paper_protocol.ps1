# FAR-Trans paper protocol (new assets only, 6-month window) on five of the paper's 61 dates,
# for a like-for-like comparison with the public benchmark in outputs/reference/far_trans_benchmark.json
$env:PYTHONWARNINGS = "ignore"
Set-Location (Split-Path $PSScriptRoot -Parent)
$dates = @(
    @("2019-11-21", "2020-05-21"), @("2020-05-22", "2020-11-22"), @("2020-11-23", "2021-05-23"),
    @("2021-05-25", "2021-11-25"), @("2021-11-23", "2022-05-23")
)
foreach ($d in $dates) {
    python src/run.py --protocol new --cutoff $d[0] --end $d[1] --baselines --tag v3 2>&1 |
        Select-String "^run |popularity_90d|ials_only|userknn_only|hybrid_fixed|reranker  "
}
python src/paper_compare.py v3
