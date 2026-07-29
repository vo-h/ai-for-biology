# `src/data/census.py`

All Census access goes through this module. Nothing is downloaded to disk — every call fetches only the requested slice via TileDB-SOMA and returns.

**Census version pinned to `2025-11-08` (LTS).** `"latest"` rotates weekly and breaks reproducibility.

---

## Functions

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
3. **QC** (thresholds from `qc_params`, default `preprocessing.QC_PARAMS`):
   - `nnz` (genes detected) and `raw_sum` (total UMI counts) against `min_genes`/`max_genes`/`min_counts`. These are **precomputed by Census over the full ~20k-gene transcriptome** and read directly off `obs` — no expression fetch needed, unlike the old scanpy-based `run_qc`.
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

`scripts/train_mlp.py` handles caching automatically (`results/mlp/hvg_list.json`).

---

## PyTorch dataset

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
from src.data.census import fetch_metadata, compute_hvg_list, build_census_dataset, CensusCollateFn
from src.data.preprocessing import get_label_encoder
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

Picklable `collate_fn` for `experiment_dataloader`. Each item from `ExperimentDataset` is `(X_ndarray, obs_df)` with raw UMI counts already restricted to the fixed HVG columns. Per batch: drops cells whose `cell_type` isn't in `label2int` (e.g. filtered out of the training label set upstream), applies `normalize_total(1e4)` + `log1p` (matching `preprocessing.normalize()`), and maps `cell_type` strings to integer labels.

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
