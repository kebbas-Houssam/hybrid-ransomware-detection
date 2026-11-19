# Datasets

No dataset files are stored in this repository (`data/*.csv` is git-ignored). Put the training CSV here, e.g. `data/full_dataset_5s.csv`, then run:

```bash
python -m training.train --data data/full_dataset_5s.csv
```

## Sources

Two public datasets with different properties were used, to test the model both in a realistic noisy setting and in a controlled one.

| Dataset | Source | Environment | Class balance | Content |
|---|---|---|---|---|
| **DeftPunk** | Wang et al., *Ransom Access Memories: Achieving Practical Ransomware Protection in Cloud with DeftPunk*, OSDI 2024 — [paper](https://www.usenix.org/conference/osdi24/presentation/wang-zhongyu) | Cloud block storage (servers) | Highly imbalanced (~7 % ransomware) | > 140 h of block-level I/O traces (33 GB), 13 ransomware families, 16 real-world workloads (databases, web servers, …) |
| **RansMap2024** (RanSMAP) | M. Hirano and R. Kobayashi, *RanSMAP: Open Dataset of Ransomware Storage and Memory Access Patterns for Creating Deep Learning Based Ransomware Detectors*, Computers & Security, 2024, 104202 — [Kaggle](https://www.kaggle.com/datasets/hiranomanabu/ransmap-2024-ransomware-behavioral-features) | Workstation (standard PC) | Balanced (~52.4 % ransomware) | Several ransomware families and standard benign applications |

Each dataset is distributed under its authors' own terms; refer to the original sources before reuse.

**Why two datasets:** DeftPunk tests the model in near-real conditions where attacks are rare ("needles in a haystack"), which stresses recall and precision. RansMap2024 measures the intrinsic discrimination ability of the architecture without the class-imbalance bias, and validates generalization.

## Preprocessing

1. The raw I/O trace stream of each sample is split into fixed **5-second windows** (5 s, 10 s and 20 s were compared; 5 s gave the best results).
2. The **13 behavioral features** below are computed for each window.
3. All windows are merged into one CSV, one row per window, with the trace identifier, window start time and label.
4. The model reads **sequences of 50 consecutive windows** (≈ 250 s of activity) per trace, with 50 % overlap.

## Expected CSV format

One row per 5-second window of storage I/O for one trace.

| Column | Type | Description |
|---|---|---|
| `file` | str | Trace / sample identifier (windows are grouped into sequences per file) |
| `start_time` | float | Window start time, used to order the windows |
| `label` | int | 1 = ransomware, 0 = benign |
| `read_write_ratio` | float | Number of reads / number of writes. A value close to 1 can reflect the read-encrypt-rewrite cycle |
| `war_ratio` | float | Write-After-Read ratio: share of writes targeting an offset already read in the same window (captures the read-encrypt-rewrite pattern) |
| `wss` | float | Working Set Size: number of unique offsets accessed |
| `entropy` | float | Shannon entropy of the distribution of accessed offsets (high = scattered accesses, e.g. file scanning) |
| `read_pct` | float | Percentage of read operations among all operations |
| `write_pct` | float | Percentage of write operations among all operations |
| `repeat_ratio` | float | Re-access rate of the same offsets: `1 - wss / total_ops` |
| `read_entropy` | float | Shannon entropy of the read offsets |
| `write_entropy` | float | Shannon entropy of the written offsets |
| `total_ops` | float | Total number of I/O operations in the window (activity intensity) |
| `write_to_unique_ratio` | float | Number of writes / number of unique offsets |
| `avg_offset_gap` | float | Mean time between two consecutive operations |
| `burstiness` | float | Standard deviation of inter-operation times (low = very regular, automated rhythm) |

The 13 feature columns (all except `file`, `start_time`, `label`) are the model input, in this order (`training/data.py::FEATURE_COLUMNS`). `predict.py` only needs the 13 features; `file` and `start_time` are optional there.
