# Split Strategies: Cross-Donor vs. Random Evaluation

**Question:** does a cell-type classifier's accuracy depend on whether the same donor's cells appear in both train and test?

## Background

Every cell shares the same genome but expresses different genes — that differential expression is what makes a T cell a T cell. Single-cell RNA-seq measures per-cell UMI counts across ~20,000 genes. The confound: expression is also shaped by **batch effects** — donor, lab, chemistry — not just cell identity. Most tutorials randomly split cells into train/test, letting the same donor appear in both. A model can then partly learn "which donor is this" as a shortcut for "what cell type is this," inflating accuracy in a way that won't hold up on a genuinely new patient.

## Method

- Data: CZ CELLxGENE Census, blood tissue, 30 donors (post-QC), 10x 3' v3, 2,000 HVGs.
- Model: `CellTypeMLP`, 2000 → 512 → 256 → n_classes, BatchNorm + Dropout.
- Two split strategies, same architecture/hyperparameters/data, 5-fold CV each, trained to convergence (`--patience 5`, up to 50 epochs):
  - **`donor`** (`group_k_fold_donors`) — disjoint donor groups per fold; no donor in both train and test.
  - **`random`** (`random_k_fold`) — random cell-level split; the same donor's cells can land in both train and test.
- External check: all 5 fold-models per split, averaged (not cherry-picked), evaluated on 3 donors entirely outside the 30-donor training pool.
- `src/evaluation/cross_donor.py`; run via `scripts/train_mlp.py --split-strategy {donor,random} --patience 5`; grouped external eval via `scripts/test_mlp.py --model-dir`.

## Results

| Split | Val macro-F1 (5-fold mean ± std) | macro-F1 on 3 fully external donors (5-model mean ± std) |
|---|---|---|
| `donor` (honest) | 0.8726 ± 0.0038 | 0.8340 ± 0.0034 |
| `random` (leaked) | 0.8760 ± 0.0086 | 0.8361 ± 0.0071 |

**The gap is still small — 0.3pp at validation time, 0.2pp externally, and well within `random`'s own fold-to-fold std.** This supersedes an earlier single-fold comparison (0.8389 vs 0.8493, ~1pp) that turned out to be a methodology artifact: that comparison picked each split's single best-validation fold, and those two folds happened to have trained for very different lengths (11 vs 50 epochs). Averaging properly across all 5 folds per split — and letting each fold train to convergence instead of a fixed epoch count — collapses that gap back down.

One loose end: `random` fold 0 hit the 50-epoch ceiling without early stopping ever triggering (still improving at the end), which is most of why `random`'s std is wider than `donor`'s here. Worth an even higher epoch cap to confirm it doesn't move the mean further.

This runs counter to the common claim that random splits inflate accuracy by 10–30pp. Best guess why: blood immune cell types are defined by strong, canonical, cross-donor-conserved marker genes (CD3/CD4/CD8/CD14/CD19/CD56...) — there isn't much donor-specific signal available for the model to exploit as a shortcut in the first place. Donor leakage should matter more in tissues with more continuous or donor-variable cell states (tumor microenvironment, brain).

## Future work

- **Rerun on a tissue with more expected batch effect** (tumor, brain) — the real test of whether the small gap is blood-specific or general.
- **More donors** — still only 30.
- **Per-class deep dive** on the persistently-hard classes (e.g. natural killer cell, F1 ≈ 0.28 in both splits) — likely annotation ambiguity rather than a modeling problem, worth checking against the cell-type label hierarchy.

## Reproduce

```bash
python scripts/download_dataset.py --tissues blood --n-donors 30 --output-dir results/donors --hvg-cache results/hvg.json
python scripts/train_mlp.py --data-dir results/donors --split-strategy donor  --patience 5 --n-epochs 50 --output-dir results/split-donors
python scripts/train_mlp.py --data-dir results/donors --split-strategy random --patience 5 --n-epochs 50 --output-dir results/split-random
python scripts/test_mlp.py --model-dir results/split-donors --data-dir results/donors-test --fname split-donors.json
python scripts/test_mlp.py --model-dir results/split-random --data-dir results/donors-test --fname split-random.json
```

## References

- Sikkema et al., *Nature Medicine* 2023 — lung atlas cross-study integration benchmark
- Luecken et al., *Nature Methods* 2022 — scRNA-seq integration benchmarking
- Heimberg et al., "Parameter-free representations outperform single-cell foundation models on downstream benchmarks" (2026 preprint)
