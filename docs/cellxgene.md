# `src/data/cellxgene.py`

QC, preprocessing, and data fetching for the CZ CELLxGENE Census, plus the two PyTorch dataset implementations trained against it. All Census access goes through this module. Nothing is downloaded to disk — every call fetches only the requested slice via TileDB-SOMA and returns — except `download_donor_h5ads`, which exists specifically to write local files that `LocalDonorDataset` then trains against without touching Census again.

**Census version pinned to `2025-11-08` (LTS).** `"latest"` rotates weekly and breaks reproducibility.

---

## Pipeline

QC and normalization are applied in two separate places, for two separate purposes:

```
compute_hvg_list (once, on a small donor-balanced sample):
  fetch_metadata (already QC'd)  →  normalize  →  select_hvg  →  fixed 2000-gene list

training / eval (every cell, every epoch):
  fetch_metadata (QC applied here)
      →  build_census_dataset (fetch only the fixed HVG columns)
          — or download_donor_h5ads → LocalDonorDataset, for the local-file path
      →  CensusCollateFn (normalize_total(1e4) + log1p, per batch)
      →  MLP
```

`run_qc` is **not called anywhere in the current pipeline**. QC now happens once in `fetch_metadata`, using Census's precomputed per-cell `nnz`/`raw_sum` plus a targeted mitochondrial-gene read (see the `fetch_metadata` caveat below for why). `run_qc` is kept in this module for exploratory/notebook use on full AnnData objects, but it plays no role in what the model actually trains on.

HVG selection (`select_hvg`) is run once upfront via `compute_hvg_list` to produce a fixed gene list. `build_census_dataset` then restricts every subsequent fetch to just that gene list — there is no per-chunk HVG-subsetting step.

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

## QC & preprocessing functions

### `run_qc`

```python
run_qc(adata, params=None) -> AnnData
```

Filters low-quality cells based on QC thresholds. Mitochondrial genes are identified by the `MT-` prefix — human-specific; change to `mt-` for mouse.

Three failure modes caught:
- **Empty droplets** — too few genes detected (`< min_genes`)
- **Doublets** — two cells captured together, so too many genes (`> max_genes`)
- **Damaged cells** — high mitochondrial fraction, indicating cytoplasmic RNA has leaked out (`> max_pct_mito`)

> **Not used in the training/eval pipeline.** `fetch_metadata` applies the equivalent checks using Census's precomputed per-cell stats instead of this function (see below). This is kept for exploratory notebooks working with an already-fetched full-gene AnnData.

---

### `normalize`

```python
normalize(adata) -> AnnData
```

1. Saves raw UMI counts to `adata.layers["counts"]` before modifying `.X`.
2. Normalises each cell to 10,000 total counts (`normalize_total`) — removes sequencing depth variation across cells.
3. Applies `log1p` — compresses the dynamic range so highly expressed genes don't dominate learning.

Raw counts are preserved in `.layers["counts"]` because HVG selection (`seurat_v3`) requires them.

Used directly (on AnnData) inside `compute_hvg_list`. The training/eval path (`CensusCollateFn`, below) re-implements the same two steps — `normalize_total(1e4)` then `log1p` — as vectorized numpy on the raw `(batch_size, n_genes)` ndarray, since `ExperimentDataset`/`LocalDonorDataset` yield ndarrays rather than AnnData and constructing an AnnData per minibatch would add unnecessary overhead in the training loop.

---

### `select_hvg`

```python
select_hvg(adata, n_top_genes=2000) -> AnnData
```

Selects the 2,000 most informative genes using the `seurat_v3` method, which fits a negative binomial model to the **raw count layer** rather than log-normalised values. Fitting to normalised values inflates variance estimates for highly expressed genes — using `layer="counts"` is current best practice.

In the training pipeline this is called once via `compute_hvg_list` to get a fixed gene list. The list is then passed to `build_census_dataset` so only those genes are ever fetched during training/eval.

---

### `get_label_encoder`

```python
get_label_encoder(labels: list[str]) -> (label2int, int2label)
```

Returns a pair of dicts for converting cell-type strings to integer class indices and back. Sorted alphabetically so the mapping is deterministic across runs and machines.

---

## Gene pre-filtering

### `fetch_candidate_genes`

```python
fetch_candidate_genes(
    min_expr_rate: float = 0.01,
    max_expr_rate: float = 0.60,
    census_version: str = CENSUS_VERSION,
) -> list[str]
```

Returns protein-coding gene names within an intermediate expression rate range, read from the corpus-wide `var` table only — no expression data fetched. Used to pre-filter genes before the HVG expression fetch, cutting the matrix from ~61k to ~13k genes before `compute_hvg_list` ever reads counts.

`expr_rate = nnz / n_measured_obs` (fraction of cells expressing the gene):
- `< min_expr_rate`: dropout genes, rarely expressed, never HVGs
- `> max_expr_rate`: constitutively expressed housekeeping genes, low variance

> **Caveat:** this filter also excludes the canonical mitochondrial genes (`MT-*`) — they're expressed in ~70–80% of cells, above the `0.60` cutoff. That's fine for HVG candidacy (mito genes are rarely useful cell-type markers), but it's why mito QC (`fetch_metadata`, below) reads MT genes through a separate, unfiltered path rather than through this candidate list.

---

## Metadata + expression fetching

### `fetch_metadata`

```python
fetch_metadata(
    tissues: list[str],
    diseases: list[str] = ["normal"],
    assays: list[str] = ["10x 3' v3"],
    min_cells_per_donor: int = 50,
    qc_params: dict = QC_PARAMS,
    census_version: str = CENSUS_VERSION,
) -> pd.DataFrame
```

Returns a cell-level metadata DataFrame for the requested slice — **this is where QC happens**. Every cell returned here is exactly what training/eval streams later; there is no further per-batch QC downstream.

Columns returned: `soma_joinid`, `tissue_general`, `cell_type`, `donor_id`, `dataset_id`, `disease`, `assay`, `sex`, `development_stage`, `raw_sum`, `nnz`, `pct_counts_mt`.

**Filters applied, in order:**
1. `is_primary_data == True` — avoids double-counting cells that appear in multiple datasets.
2. `disease` post-filtered with regex — the field can contain multiple values delimited by ` || `; exact equality silently misses these rows.
3. **QC** (thresholds from `qc_params`, default `QC_PARAMS` above):
   - `nnz` (genes detected) and `raw_sum` (total UMI counts) against `min_genes`/`max_genes`/`min_counts`. These are **precomputed by Census over the full ~20k-gene transcriptome** and read directly off `obs` — no expression fetch needed, unlike the scanpy-based `run_qc`.
   - `pct_counts_mt` against `max_pct_mito`, computed from a targeted read of just the ~37 `MT-`-prefixed gene columns (see caveat below) — cheap relative to fetching all genes.
4. Donors with fewer than `min_cells_per_donor` cells **after QC** are dropped.

> **Mito QC caveat:** ~8% of Census cells belong to datasets that never measured any MT- gene at all (e.g. targeted probe panels). Reading MT genes for those cells returns zero rows, which would look identical to a genuinely clean 0%-mito cell. `fetch_metadata` cross-checks against Census's dataset presence matrix (`cellxgene_census.get_presence_matrix`) and treats "never measured" as a QC failure (dropped), not an automatic pass.

**`assays` accepts multiple values** — useful when pooling compatible chemistries (e.g. `10x 3' v2` + `10x 3' v3`) to increase data volume without mixing fundamentally different protocols.

**Example:**
```python
# Single assay
meta = fetch_metadata(tissues=["blood", "lung"])

# Pooling two 10x chemistries
meta = fetch_metadata(
    tissues=["blood"],
    assays=["10x 3' v2", "10x 3' v3"],
)
```

---

## HVG list computation

### `compute_hvg_list`

```python
compute_hvg_list(
    tissues: list[str],
    n_sample_cells: int = 20_000,
    n_hvg: int = 2000,
    census_version: str = CENSUS_VERSION,
) -> list[str]
```

Computes a fixed HVG gene list from a small representative sample. **Call once, cache to disk, pass to every `build_census_dataset` call.** All training/eval cells must use the same gene list — if each split computed its own HVGs the feature vectors would be incompatible across them.

Sampling is done evenly across donors so HVG selection isn't dominated by the expression profile of any single high-cell-count donor. Cells come from `fetch_metadata`, so they're already QC-passed — this function only normalizes (`normalize_total` + `log1p`) and runs `select_hvg` (`seurat_v3` on raw counts); it does not re-run QC.

`scripts/cellxgene/train_mlp.py` handles caching automatically (`results/mlp/hvg_list.json`).

---

## PyTorch dataset (live Census streaming)

### `build_census_dataset`

```python
build_census_dataset(
    census,
    soma_joinids: list[int],
    hvg_genes: list[str],
    batch_size: int = 256,
    io_batch_size: int = 65_536,
    shuffle_chunk_size: int = 64,
    shuffle: bool = True,
    organism: str = "Homo sapiens",
) -> ExperimentDataset
```

Builds a `tiledbsoma_ml.ExperimentDataset` (a PyTorch `IterableDataset`) restricted to the given `soma_joinids` (a donor-split cell population) and `hvg_genes`. Takes an already-open `census` handle — the caller controls context lifetime, since worker processes reopen the underlying array independently via a stored URI (not a live reference to `census`), so `census` only needs to stay open long enough for `build_census_dataset` itself to run, not for the full duration of training.

`shuffle_chunk_size` / `io_batch_size` control shuffle granularity vs. I/O locality — see the [tiledbsoma_ml docs](https://github.com/single-cell-data/TileDB-SOMA-ML) for the shuffle-chunking → IO-batching → mini-batching pipeline. `shuffle_chunk_size` is ignored when `shuffle=False`.

**Example:**
```python
from src.data.cellxgene import fetch_metadata, compute_hvg_list, build_census_dataset, CensusCollateFn, get_label_encoder
from tiledbsoma_ml import experiment_dataloader
import cellxgene_census

hvg_genes = compute_hvg_list(tissues=["blood"])          # once, then cache
meta      = fetch_metadata(tissues=["blood"])             # already QC'd
label2int, _ = get_label_encoder(meta["cell_type"].tolist())
collate = CensusCollateFn(label2int)

with cellxgene_census.open_soma() as census:
    ds = build_census_dataset(
        census,
        soma_joinids=meta["soma_joinid"].tolist(),
        hvg_genes=hvg_genes,
        batch_size=256,
    )
    loader = experiment_dataloader(ds, num_workers=2, collate_fn=collate)
    for X, y in loader:
        # X: (256, 2000) float32 tensor, X_batch already log1p-normalized by collate
        # y: (256,) long tensor of class indices
        ...
```

### `CensusCollateFn`

```python
CensusCollateFn(label2int: dict[str, int])
```

Picklable `collate_fn`, shared by `experiment_dataloader` (Census streaming) and the plain `DataLoader` used with `LocalDonorDataset` (below) — both yield the same `(X_ndarray, obs_df)` item shape. Each item has raw UMI counts already restricted to the fixed HVG columns. Per batch: drops cells whose `cell_type` isn't in `label2int` (e.g. filtered out of the training label set upstream), applies `normalize_total(1e4)` + `log1p` (matching `normalize()` above), and maps `cell_type` strings to integer labels.

---

## Download to local files

### `download_donor_h5ads`

```python
download_donor_h5ads(
    meta: pd.DataFrame,
    hvg_genes: list[str],
    output_dir: Path,
    organism: str = "Homo sapiens",
    census_version: str = CENSUS_VERSION,
) -> list[Path]
```

Fetches cells donor-by-donor and writes one `{donor_id}.h5ad` file per donor to `output_dir`. Each donor is fetched and written independently, so memory use is bounded by a single donor's slice — never the full `meta` population. `meta` must already be QC'd (see `fetch_metadata`) and carry `soma_joinid`/`donor_id`/`cell_type`. All donor files share the same fixed HVG gene columns, in the same order, so they stay directly concatenable/comparable downstream (e.g. `anndata.experimental.AnnCollection` or `concat_on_disk`).

Idempotent / resumable: a donor whose file already exists is skipped, so an interrupted run can be picked back up by just rerunning with the same `output_dir`. Each file is written to a `.tmp` sibling first and atomically renamed into place on success — a crash mid-write leaves only a stray `.tmp` file, never a truncated `.h5ad` that a resume would mistake for already-downloaded.

Driven by `scripts/cellxgene/download_dataset.py`.

---

## PyTorch dataset (local per-donor h5ad files)

Trains directly against files written by `download_donor_h5ads`, without touching Census again — used by `scripts/cellxgene/train_mlp.py --data-dir` and `scripts/cellxgene/test_mlp.py`.

### `list_available_donors`

```python
list_available_donors(data_dir: Path) -> list[str]
```

Returns `donor_id`s for every `{donor_id}.h5ad` file in `data_dir`.

### `LocalDonorDataset`

```python
LocalDonorDataset(
    data_dir: Path,
    donor_ids: list[str] | None = None,
    cell_indices: dict[str, np.ndarray] | None = None,
    batch_size: int = 256,
    shuffle: bool = True,
    seed: int = 0,
)
```

A `torch.utils.data.IterableDataset` that streams `(X_batch, obs_batch)` mini-batches from local per-donor h5ad files. Donors are the unit of both storage and memory: only one donor's h5ad is opened and materialized at a time — its cells are shuffled in-memory (if `shuffle`) and split into `batch_size` mini-batches before the next donor is opened. Even that donor's `X` is only densified per mini-batch (not all at once), since Census raw counts read back as sparse.

Yields the same `(X_ndarray, obs_df)` shape as `ExperimentDataset` above, so `CensusCollateFn` works unchanged as the `collate_fn`. Wrap in a plain `torch.utils.data.DataLoader` (not `tiledbsoma_ml`'s `experiment_dataloader`, which is Census-specific), with `batch_size=None` since batching is handled internally.

By default every cell in each listed donor's file is used. Pass `cell_indices` (`donor_id` → row positions within that donor's file) to restrict to a specific subset of cells per donor instead — needed for splits that cut across donor boundaries (e.g. a random cell-level split, as opposed to the whole-donor splits cross-donor CV uses).

Call `set_epoch(epoch)` before each epoch to reseed donor visit order and in-donor shuffling. With multiple `DataLoader` workers, donors are partitioned across workers (`donor_ids[worker.id::worker.num_workers]`) so each worker streams a disjoint subset rather than redundantly re-streaming every donor.

---

## Design notes

| Decision | Reason |
|----------|--------|
| `assay` filter uses double quotes inside the expression string | The assay name `10x 3' v3` contains a literal single quote — single-quoting the value breaks the TileDB-SOMA query parser. |
| `disease` post-filtered with `str.contains` after fetch | The `value_filter` string is evaluated by TileDB-SOMA; regex functions are not available in that context. |
| QC uses Census's precomputed `nnz`/`raw_sum` instead of scanpy's `calculate_qc_metrics` | Those two stats are already computed by CZI over the full transcriptome and exposed on `obs` — free to read, and more accurate than computing them on any gene-subsetted fetch (e.g. the HVG candidate pool, which already excludes high-expression genes like the mito set). |
| Mito QC reads MT- genes directly from `var`, not from `fetch_candidate_genes`'s output | Mitochondrial genes are expressed in the large majority of cells, so `fetch_candidate_genes`'s `max_expr_rate=0.60` cutoff excludes nearly all of them — computing `pct_counts_mt` from the candidate/HVG gene pool would silently read as ~0% for almost every cell. |
| Datasets with zero MT genes measured are dropped, not treated as 0% mito | ~8% of Census cells come from datasets (e.g. targeted probe panels) that never captured MT- genes. Treating "not measured" the same as "measured, zero counts" would let damaged/dying cells from those datasets pass QC unconditionally. |
| QC happens once in `fetch_metadata`, not per training batch | `build_census_dataset` fetches only the fixed HVG columns (~2,000 of ~20k genes) to keep I/O light — by the time a training batch is read, there isn't enough of the transcriptome left to compute `nnz`/`pct_counts_mt` correctly. Filtering upfront, on the full `obs` metadata (which already carries full-transcriptome `nnz`/`raw_sum`), keeps the cheap I/O path while still applying real QC. |
| `ExperimentDataset` reopens the array via a stored URI, not the original `axis_query` object | `build_census_dataset` closes its `axis_query` context manager on return — this is safe because `tiledbsoma_ml`'s `ExperimentDataset` only pulls `obs`/`var` joinids and an `XLocator` (URI + tiledb config) out of the query at construction time; it does not hold a live reference to it. |
| HVG selection on raw counts (`layer="counts"`) | `seurat_v3` assumes count data; log-normalised values violate the model assumptions and bias gene selection toward highly expressed genes. |
| Raw counts saved to `.layers["counts"]` | `seurat_v3` HVG selection reads this layer explicitly — `.X` is overwritten by `normalize_total`. |
| 2,000 HVGs | Standard for 10x data. Captures ~95% of biological variance while reducing feature dimensionality ~10x from the full gene set. |
| No PCA step | The MLP's first linear layer is strictly more expressive than PCA — it learns a nonlinear projection tuned to cell-type discrimination. BatchNorm takes over the standardisation role that PCA preprocessing used to serve. |
| `LocalDonorDataset` materializes one donor at a time, never the full directory | Bounds memory to the largest single donor rather than the whole downloaded population, at the cost of shuffling only within a donor, not across donors globally — acceptable since `download_donor_h5ads` output is already meant to be read multiple epochs by a dataset that shuffles donor visit order per epoch. |
