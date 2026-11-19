# Datasets

No dataset files are stored in this repository (`data/*.csv` is git-ignored).
Put the training CSV here, e.g. `data/full_dataset_5s.csv`, then run
`python -m training.train --data data/full_dataset_5s.csv`.

## Sources

| Dataset | Source | License | Notes |
|---|---|---|---|
| DeftPunk | _TODO: URL / citation_ | _TODO_ | _TODO: imbalanced; families, number of traces_ |
| RansMap2024 | _TODO: URL / citation_ | _TODO_ | _TODO: balanced; families, number of traces_ |

_TODO: describe how the raw I/O traces were turned into 5-second windows and merged into one CSV._

## Expected CSV format

One row per 5-second window of storage I/O for one trace (`file`).

| Column | Type | Description |
|---|---|---|
| `file` | str | Trace / sample identifier (windows are grouped into sequences per file) |
| `start_time` | float | Window start time, used to order the windows |
| `label` | int | `1` = ransomware, `0` = benign |
| `read_write_ratio` | float | _TODO_ |
| `war_ratio` | float | _TODO: write-after-read ratio_ |
| `wss` | float | _TODO: working set size_ |
| `entropy` | float | _TODO_ |
| `read_pct` | float | _TODO_ |
| `write_pct` | float | _TODO_ |
| `repeat_ratio` | float | _TODO_ |
| `read_entropy` | float | _TODO_ |
| `write_entropy` | float | _TODO_ |
| `total_ops` | float | _TODO_ |
| `write_to_unique_ratio` | float | _TODO_ |
| `avg_offset_gap` | float | _TODO_ |
| `burstiness` | float | _TODO_ |

The 13 feature columns (all except `file`, `start_time`, `label`) are the model input,
in this order (`training/data.py::FEATURE_COLUMNS`). `predict.py` only needs the 13
features; `file` and `start_time` are optional there.
