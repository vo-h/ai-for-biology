# Codebase Overview

```
src/
  data/
    cellxgene.py        # Census fetch + QC, HVG computation, ExperimentDataset builder,
                         # LocalDonorDataset (per-donor h5ad training) — see cellxgene.md
    rxrx1.py             # RxRx1 GCS metadata fetch, RxRx1Dataset, RxRx1CollateFn — see rxrx1.md
  models/
    mlp.py              # CellTypeMLP: Linear → BatchNorm → ReLU → Dropout
  training/
    cellxgene.py        # train_epoch, eval_epoch, run_training, model_config.json
    callbacks.py         # EarlyStopping
    run_metadata.py       # hardware/timing captured per run
  evaluation/
    cellxgene.py         # group_k_fold_donors, random_k_fold, leave_one_donor_out,
                          # macro_f1, confusion matrix, accuracy gap report
scripts/
  cellxgene/
    download_dataset.py  # download a QC'd Census slice to local per-donor h5ad files
    train_mlp.py          # CLI training entry point (--split-strategy donor|random)
    test_mlp.py            # evaluate a saved checkpoint against a local h5ad directory
  rxrx1/
    test_dataloader.py    # RxRx1Dataset throughput benchmark
  sync.sh                   # push code / pull results to a cloud training instance
```

- [cellxgene.md](cellxgene.md) — Census data fetching, QC (incl. mito-QC caveats), preprocessing, HVG computation, and both PyTorch dataset implementations (live streaming + local h5ad files).
- [rxrx1.md](rxrx1.md) — RxRx1 bucket layout, metadata fetching, channel-stat normalization, and the live-streaming PyTorch dataset (incl. why `fs.cat()` and `worker_init_fn` matter).

See `projects/` for what a specific experiment does with this.
