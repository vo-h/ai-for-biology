# Codebase Overview

```
src/
  data/
    census.py          # Census fetch + QC, HVG computation, ExperimentDataset builder — see census.md
    local.py            # LocalDonorDataset — train from downloaded per-donor h5ad files
    preprocessing.py    # run_qc (exploratory), normalize, select_hvg — see preprocessing.md
  models/
    mlp.py              # CellTypeMLP: Linear → BatchNorm → ReLU → Dropout
  training/
    trainer.py          # train_epoch, eval_epoch, run_training, model_config.json
    callbacks.py         # EarlyStopping
    run_metadata.py       # hardware/timing captured per run
  evaluation/
    cross_donor.py       # group_k_fold_donors, random_k_fold, leave_one_donor_out
    metrics.py            # macro_f1, confusion matrix, accuracy gap report
scripts/
  download_dataset.py    # download a QC'd Census slice to local per-donor h5ad files
  train_mlp.py            # CLI training entry point (--split-strategy donor|random)
  test_mlp.py              # evaluate a saved checkpoint against a local h5ad directory
  test.py                  # ExperimentDataset throughput benchmark
  sync.sh                   # push code / pull results to a cloud training instance
```

- [census.md](census.md) — Census data fetching, QC (incl. mito-QC caveats), HVG computation, dataset builders.
- [preprocessing.md](preprocessing.md) — QC/normalize/HVG functions used on samples.

See `projects/` for what a specific experiment does with this.
