# `src/data/preprocessing.py`

Standard scanpy preprocessing for scRNA-seq data, used in two different places for two different purposes — see the pipeline breakdown below. Both paths apply the same treatment to every cell so cells are comparable regardless of where they entered the pipeline.

---

## Pipeline

QC and normalization are **not** both applied in the same place anymore. Two separate paths:

```
compute_hvg_list (once, on a small donor-balanced sample):
  fetch_metadata (already QC'd)  →  normalize  →  select_hvg  →  fixed 2000-gene list

training / eval (every cell, every epoch):
  fetch_metadata (QC applied here, via src/data/census.py — see docs/census.md)
      →  build_census_dataset (fetch only the fixed HVG columns)
      →  CensusCollateFn (normalize_total(1e4) + log1p, per batch)
      →  MLP
```

`run_qc` in this module is **not called anywhere in the current pipeline**. QC now happens once in `census.fetch_metadata`, using Census's precomputed per-cell `nnz`/`raw_sum` plus a targeted mitochondrial-gene read — see `docs/census.md` for why (short version: by the time training data is fetched, only the fixed ~2,000 HVG columns exist, which isn't enough of the transcriptome to compute QC metrics like `pct_counts_mt` correctly). `run_qc` is kept in this module for exploratory/notebook use on full AnnData objects, but it plays no role in what the model actually trains on.

HVG selection (`select_hvg`) is run once upfront via `compute_hvg_list` to produce a fixed gene list. `build_census_dataset` (in `census.py`) then restricts every subsequent fetch to just that gene list — this module has no per-chunk HVG-subsetting step.

---

## Defaults

```python
QC_PARAMS = {
    "min_genes": 200,      # below → likely empty droplet
    "max_genes": 6000,     # above → likely doublet (two cells captured in one droplet)
    "max_pct_mito": 20.0,  # above → likely damaged or dying cell
    "min_counts": 500,     # total UMI counts per cell
}

N_HVG = 2000   # genes in the fixed feature vector
```

---

## Functions

### `run_qc`

```python
run_qc(adata, params=None) -> AnnData
```

Filters low-quality cells based on QC thresholds. Mitochondrial genes are identified by the `MT-` prefix — human-specific; change to `mt-` for mouse.

Three failure modes caught:
- **Empty droplets** — too few genes detected (`< min_genes`)
- **Doublets** — two cells captured together, so too many genes (`> max_genes`)
- **Damaged cells** — high mitochondrial fraction, indicating cytoplasmic RNA has leaked out (`> max_pct_mito`)

> **Not used in the training/eval pipeline.** `census.fetch_metadata` applies the equivalent checks using Census's precomputed per-cell stats instead of this function (see `docs/census.md`). This is kept for exploratory notebooks working with an already-fetched full-gene AnnData.

---

### `normalize`

```python
normalize(adata) -> AnnData
```

1. Saves raw UMI counts to `adata.layers["counts"]` before modifying `.X`.
2. Normalises each cell to 10,000 total counts (`normalize_total`) — removes sequencing depth variation across cells.
3. Applies `log1p` — compresses the dynamic range so highly expressed genes don't dominate learning.

Raw counts are preserved in `.layers["counts"]` because HVG selection (`seurat_v3`) requires them.

Used directly (on AnnData) inside `compute_hvg_list`. The training/eval path (`census.CensusCollateFn`) re-implements the same two steps — `normalize_total(1e4)` then `log1p` — as vectorized numpy on the raw `(batch_size, n_genes)` ndarray, since `ExperimentDataset` yields ndarrays rather than AnnData and constructing an AnnData per minibatch would add unnecessary overhead in the training loop.

---

### `select_hvg`

```python
select_hvg(adata, n_top_genes=2000) -> AnnData
```

Selects the 2,000 most informative genes using the `seurat_v3` method, which fits a negative binomial model to the **raw count layer** rather than log-normalised values. Fitting to normalised values inflates variance estimates for highly expressed genes — using `layer="counts"` is current best practice.

In the training pipeline this is called once via `compute_hvg_list` to get a fixed gene list. The list is then passed to `build_census_dataset` (see `docs/census.md`) so only those genes are ever fetched during training/eval.

---

### `get_label_encoder`

```python
get_label_encoder(labels: list[str]) -> (label2int, int2label)
```

Returns a pair of dicts for converting cell-type strings to integer class indices and back. Sorted alphabetically so the mapping is deterministic across runs and machines.

---

## Design notes

| Decision | Reason |
|----------|--------|
| HVG selection on raw counts (`layer="counts"`) | `seurat_v3` assumes count data; log-normalised values violate the model assumptions and bias gene selection toward highly expressed genes. |
| Raw counts saved to `.layers["counts"]` | `seurat_v3` HVG selection reads this layer explicitly — `.X` is overwritten by `normalize_total`. |
| 2,000 HVGs | Standard for 10x data. Captures ~95% of biological variance while reducing feature dimensionality ~10x from the full gene set. |
| No PCA step | The MLP's first linear layer is strictly more expressive than PCA — it learns a nonlinear projection tuned to cell-type discrimination. BatchNorm takes over the standardisation role that PCA preprocessing used to serve. |
