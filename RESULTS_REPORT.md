# Rankings CSV Report

## Scope
This report analyzes only `outputs/rankings.csv`.

## File Summary
- Rows: `150`
- Columns: `6`
- Columns present: `job_id`, `job_title`, `rank`, `candidate_id`, `similarity`, `explanation`

## Coverage
- Unique jobs: `50`
- Unique candidates: `100`
- Rows per job: min `3`, mean `3.0`, max `3`
- Rows per candidate: min `1`, mean `1.5`, max `7`

Interpretation:
- The file is a strict Top-K output with `K=3` for every job.

## Rank Distribution
- Rank 1: `50`
- Rank 2: `50`
- Rank 3: `50`

Interpretation:
- Ranking depth is consistent across all jobs (no missing or extra rank levels).

## Similarity Score Analysis
- Minimum: `0.359437`
- 25th percentile: `0.481428`
- Median: `0.531788`
- Mean: `0.539347`
- 75th percentile: `0.592663`
- Maximum: `0.702428`
- Standard deviation: `0.073622`

Interpretation:
- Scores are concentrated in a moderate-to-strong similarity range.
- Spread is controlled (std ~`0.074`), indicating stable ranking behavior in this batch.

## Explanation Quality Check
- Non-empty explanations: `150`
- Missing/blank explanations: `0`
- Coverage: `100%`

Interpretation:
- Every ranked row includes explanation text.

## Top Similar Matches (by score)
1. `jd_101360` | Senior Human Resources Manager | `cand_0037` | rank `1` | sim `0.702428`
2. `jd_104876` | Outside Sales Representative | `cand_1061` | rank `1` | sim `0.693703`
3. `jd_67816` | Clinical Director | `cand_0511` | rank `1` | sim `0.679202`
4. `jd_23633` | ACCOUNT DIRECTOR I-ENTERPRISE 1 1 | `cand_2261` | rank `1` | sim `0.671891`
5. `jd_106600` | Inventory Control Supervisor, Waikiki | `cand_0843` | rank `1` | sim `0.671314`

## Conclusion
`rankings.csv` is internally consistent and complete:
1. Exact Top-3 structure per job is preserved.
2. Similarity scores are plausible and well-distributed.
3. Explanations are present for all rows.
