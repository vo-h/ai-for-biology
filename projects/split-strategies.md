# Split Strategies: Cross-Donor vs. Random Evaluation

**Question:** does a cell-type classifier's accuracy depend on whether the same donor's cells appear in both train and test?

## Background

Every cell shares the same genome but expresses different genes — that differential expression is what makes a T cell a T cell. Single-cell RNA-seq measures per-cell UMI counts across ~20,000 genes. The confound: expression is also shaped by **batch effects** — donor, lab, chemistry — not just cell identity. Most tutorials randomly split cells into train/test, letting the same donor appear in both. A model can then partly learn "which donor is this" as a shortcut for "what cell type is this," inflating accuracy in a way that won't hold up on a genuinely new patient.

## Method

- Data: CZ CELLxGENE Census, blood tissue, 30 donors (post-QC), 10x 3' v3, 2,000 HVGs.
- Model: `CellTypeMLP`, 2000 → 512 → 256 → n_classes, BatchNorm + Dropout.
- Two split strategies, same architecture/hyperparameters/data, 5-fold CV each:
  - **`donor`** (`group_k_fold_donors`) — disjoint donor groups per fold; no donor in both train and test.
  - **`random`** (`random_k_fold`) — random cell-level split; the same donor's cells can land in both train and test.
- `src/evaluation/cross_donor.py`; run via `scripts/train_mlp.py --split-strategy {donor,random}`.

## Results

| Split | Val macro-F1 (5-fold mean ± std) | macro-F1 on 3 fully external donors |
|---|---|---|
| `donor` (honest) | 0.8845 ± 0.0027 | 0.8467 |
| `random` (leaked) | 0.8882 ± 0.0014 | 0.8469 |

**The gap is small** — 0.4pp at validation time, ~0pp against 3 donors entirely outside the 30-donor training pool (fold-4 models, evaluated via `scripts/test_mlp.py`). Per-class F1 tells the same story: differences between the two splits are ±0.01–0.03 and go in both directions with no systematic pattern favoring `random` — this isn't a macro-F1 averaging artifact hiding a real gap.

This runs counter to the common claim that random splits inflate accuracy by 10–30pp. Best guess why: blood immune cell types are defined by strong, canonical, cross-donor-conserved marker genes (CD3/CD4/CD8/CD14/CD19/CD56...) — there isn't much donor-specific signal available for the model to exploit as a shortcut in the first place. Donor leakage should matter more in tissues with more continuous or donor-variable cell states (tumor microenvironment, brain).

## Future work

- **Rerun on a tissue with more expected batch effect** (tumor, brain) — the real test of whether the small gap is blood-specific or general.
- **More donors / more epochs** — only 30 donors and 5 epochs so far; more training could give the `random` model more opportunity to actually learn donor identity as a shortcut.
- **More replicates** — one seed per split condition; repeat with multiple seeds for a real confidence interval on the gap.
- **Per-class deep dive** on the persistently-hard classes (e.g. natural killer cell, F1 ≈ 0.28 in both splits) — likely annotation ambiguity rather than a modeling problem, worth checking against the cell-type label hierarchy.

## Reproduce

```bash
python scripts/download_dataset.py --tissues blood --n-donors 30 --output-dir results/donors --hvg-cache results/hvg.json
python scripts/train_mlp.py --data-dir results/donors --split-strategy donor  --output-dir results/split-donors
python scripts/train_mlp.py --data-dir results/donors --split-strategy random --output-dir results/split-random
python scripts/test_mlp.py --model-path results/split-donors/best_model_fold4.pt --data-dir results/donors-test
```

## References

- Sikkema et al., *Nature Medicine* 2023 — lung atlas cross-study integration benchmark
- Luecken et al., *Nature Methods* 2022 — scRNA-seq integration benchmarking
- Heimberg et al., "Parameter-free representations outperform single-cell foundation models on downstream benchmarks" (2026 preprint)
