"""
CZ CELLxGENE Census — data fetching and PyTorch dataset.

All Census access goes through this module. Every call fetches only the
requested slice via TileDB-SOMA and returns — nothing is downloaded to disk,
except download_donor_h5ads, which exists specifically to write local files.

Census version pinned to 2025-11-08 (LTS). Pin explicitly; "latest" rotates
weekly and breaks reproducibility.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import cellxgene_census
import numpy as np
import pandas as pd
import tiledbsoma
import torch
from tiledbsoma_ml import ExperimentDataset
from tqdm import tqdm

from src.data.preprocessing import QC_PARAMS, normalize, select_hvg

CENSUS_VERSION = "2025-11-08"

# Tuned TileDB context: parallel S3 ops + larger multipart parts.
# Shared across all open_soma calls so the fragment metadata is loaded once
# per context and cached in memory for the lifetime of the context.
#
# vfs.read_ahead_cache_size raised from the 10MB core default: training runs
# the same donors' cells against the same fixed HVG columns for n_epochs, so
# the same tiles get re-touched every epoch. A cache this small evicted
# almost immediately; sized up so one fold's working set can survive across
# epochs instead of re-fetching from S3 each time (measured with
# tiledbsoma_stats: touching 2000 scattered HVG columns out of 61497 forces
# ~6x more tiles read than the same column count contiguous, so cache misses
# here are expensive).
SOMA_CTX = tiledbsoma.SOMATileDBContext(tiledb_config={
    "vfs.s3.region": "us-west-2",
    "vfs.s3.max_parallel_ops": "8",
    "vfs.s3.multipart_part_size": "104857600",  # 100 MB parts
    "sm.io_concurrency_level": "8",
    "vfs.read_ahead_cache_size": "1073741824",  # 1 GB, up from 10MB default
    "sm.mem.tile_upper_memory_limit": "2147483648",  # 2 GB, up from 1GB default
})


# ---------------------------------------------------------------------------
# Gene pre-filtering from corpus-wide var stats
# ---------------------------------------------------------------------------

def fetch_candidate_genes(
    min_expr_rate: float = 0.01,
    max_expr_rate: float = 0.60,
    census_version: str = CENSUS_VERSION,
) -> list[str]:
    """
    Return protein-coding gene names within an intermediate expression rate range.

    Reads only the corpus-wide var table — no expression data fetched.
    Used to pre-filter genes before the HVG expression fetch, cutting the
    matrix from ~61k to ~13k genes.

    expr_rate = nnz / n_measured_obs (fraction of cells expressing the gene):
    - < min_expr_rate: dropout genes, rarely expressed, never HVGs
    - > max_expr_rate: constitutively expressed housekeeping genes, low variance
    """
    with cellxgene_census.open_soma(census_version=census_version, context=SOMA_CTX) as census:
        var = (
            census["census_data"]["homo_sapiens"]["ms"]["RNA"]["var"]
            .read()
            .concat()
            .to_pandas()
        )
    var["expr_rate"] = var["nnz"] / var["n_measured_obs"]
    mask = (
        (var["feature_type"] == "protein_coding")
        & (var["expr_rate"] > min_expr_rate)
        & (var["expr_rate"] < max_expr_rate)
    )
    return var.loc[mask, "feature_name"].tolist()


# ---------------------------------------------------------------------------
# Metadata + expression fetching
# ---------------------------------------------------------------------------

def fetch_mito_pct(census, organism_key: str, obs: pd.DataFrame) -> pd.Series:
    """
    Return pct-mito (0-100) per soma_joinid, indexed like `obs`.

    Reads only the ~37 "MT-"-prefixed gene columns rather than the full ~20k-gene
    matrix — total UMI counts (`raw_sum`) are already precomputed by Census per cell,
    so only the mitochondrial-gene numerator needs an expression fetch.

    ~8% of Census cells belong to datasets that never measured any MT- gene (e.g.
    targeted probe panels) — reading those returns zero rows, which would silently
    look like a clean 0% mito cell. Checked against the dataset presence matrix and
    returned as NaN instead, so those cells fail the QC comparison (NaN comparisons
    are always False) rather than auto-passing.
    """
    var = (
        census["census_data"][organism_key]["ms"]["RNA"]["var"]
        .read(column_names=["soma_joinid", "feature_name"])
        .concat()
        .to_pandas()
    )
    mt_joinids = var.loc[var["feature_name"].str.startswith("MT-"), "soma_joinid"].to_numpy()

    presence = cellxgene_census.get_presence_matrix(census, "Homo sapiens", "RNA")
    datasets = (
        census["census_info"]["datasets"]
        .read(column_names=["soma_joinid", "dataset_id"])
        .concat()
        .to_pandas()
    )
    has_mt = np.asarray(presence[:, mt_joinids].sum(axis=1)).ravel() > 0
    datasets_with_mt = set(datasets.loc[has_mt, "dataset_id"])

    obs = obs.set_index("soma_joinid")
    mt_measured = obs["dataset_id"].isin(datasets_with_mt)

    X = census["census_data"][organism_key]["ms"]["RNA"]["X"]["raw"]
    measured_joinids = obs.index[mt_measured].tolist()
    tbl = X.read(coords=(measured_joinids, mt_joinids.tolist())).tables().concat().to_pandas()
    mt_sum = tbl.groupby("soma_dim_0")["soma_data"].sum()

    pct = pd.Series(np.nan, index=obs.index)
    pct.loc[mt_measured] = (
        mt_sum.reindex(obs.index[mt_measured], fill_value=0.0) / obs.loc[mt_measured, "raw_sum"] * 100
    )
    return pct


def fetch_metadata(
    tissues: list[str],
    diseases: list[str] = ["normal"],
    assays: list[str] = ["10x 3' v3"],
    min_cells_per_donor: int = 50,
    qc_params: dict = QC_PARAMS,
    census_version: str = CENSUS_VERSION,
) -> pd.DataFrame:
    """
    Return a cell-level metadata DataFrame for the requested slice.

    Columns: soma_joinid, tissue_general, cell_type, donor_id, dataset_id,
    disease, assay, sex, development_stage, raw_sum, nnz, pct_counts_mt.

    Filters applied beyond the raw query:
    - is_primary_data == True  (avoids double-counting multi-tissue datasets)
    - disease post-filtered with regex (field can contain ' || '-delimited values)
    - QC (see `qc_params`, default `preprocessing.QC_PARAMS`): cells outside
      min/max genes-detected or min total counts are dropped using Census's
      precomputed per-cell `nnz`/`raw_sum` (full-transcriptome stats, no expression
      fetch needed); cells above `max_pct_mito` are dropped using a targeted read
      of the MT- gene columns. Cells whose dataset never measured any MT- gene
      (~8% of Census, e.g. targeted probe panels) can't have mito QC assessed and
      are dropped too, rather than silently passing as 0% mito. This is the only
      QC applied — every cell returned here is what training/eval streams, so no
      further per-batch QC is needed.
    - donors with < min_cells_per_donor cells (after QC) are dropped
    """
    tissue_filter = " or ".join(f"tissue_general == '{t}'" for t in tissues)
    assay_filter = " or ".join(f'assay == "{a}"' for a in assays)
    disease_pattern = "|".join(re.escape(d) for d in diseases)
    organism_key = "homo_sapiens"

    with cellxgene_census.open_soma(census_version=census_version, context=SOMA_CTX) as census:
        obs = (
            census["census_data"][organism_key]
            .obs.read(
                value_filter=f"({tissue_filter}) and ({assay_filter})",
                column_names=[
                    "soma_joinid", "tissue_general", "cell_type", "donor_id",
                    "dataset_id", "disease", "assay", "sex",
                    "development_stage", "is_primary_data", "raw_sum", "nnz",
                ],
            )
            .concat()
            .to_pandas()
        )

        obs = obs[obs["is_primary_data"]].copy()
        obs = obs[obs["disease"].str.contains(disease_pattern, na=False)].copy()

        obs = obs[
            (obs["nnz"] >= qc_params["min_genes"])
            & (obs["nnz"] <= qc_params["max_genes"])
            & (obs["raw_sum"] >= qc_params["min_counts"])
        ].copy()

        pct_counts_mt = fetch_mito_pct(census, organism_key, obs)
        obs = obs.set_index("soma_joinid")
        obs["pct_counts_mt"] = pct_counts_mt
        obs = obs[obs["pct_counts_mt"] <= qc_params["max_pct_mito"]].reset_index()

    donor_counts = obs["donor_id"].value_counts()
    obs = obs[obs["donor_id"].isin(donor_counts[donor_counts >= min_cells_per_donor].index)].copy()

    return obs.reset_index(drop=True)


# ---------------------------------------------------------------------------
# HVG list computation
# ---------------------------------------------------------------------------

def compute_hvg_list(
    tissues: list[str],
    n_sample_cells: int = 5_000,
    n_hvg: int = 2000,
    census_version: str = CENSUS_VERSION,
) -> list[str]:
    """
    Compute a fixed HVG list from a small representative sample.

    Sampled evenly across donors so no single high-cell-count donor dominates
    the variance estimates. A single open_soma context is held open across all
    chunk fetches so the TileDB fragment metadata is loaded once, not once per
    chunk. Call once, cache to disk, pass to build_census_dataset.
    """
    t0 = time.time()
    print("  [1/5] Fetching candidate genes from var table...")
    candidate_genes = fetch_candidate_genes(census_version=census_version)
    print(f"       {len(candidate_genes):,} protein-coding genes with intermediate expression rate  ({time.time()-t0:.0f}s)")

    t1 = time.time()
    print("  [2/5] Fetching metadata...")
    meta = fetch_metadata(tissues=tissues, census_version=census_version)
    per_donor = max(1, n_sample_cells // meta["donor_id"].nunique())
    sampled = pd.concat([
        grp.sample(min(len(grp), per_donor), random_state=42)
        for _, grp in meta.groupby("donor_id", observed=True)
    ]).head(n_sample_cells).reset_index(drop=True)
    print(f"       {len(sampled):,} cells sampled from {sampled['donor_id'].nunique()} donors  ({time.time()-t1:.0f}s)")

    t2 = time.time()
    joinids = sorted(sampled["soma_joinid"].tolist())
    print(f"  [3/5] Fetching expression for {len(joinids):,} cells × {len(candidate_genes):,} genes in one query...")
    with cellxgene_census.open_soma(census_version=census_version, context=SOMA_CTX) as census:
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="Homo sapiens",
            obs_coords=joinids,
            var_value_filter=f"feature_name in {candidate_genes}",
            obs_column_names=["soma_joinid", "cell_type"],
            var_column_names=["feature_name"],
        )
    print(f"       {adata.n_obs:,} cells × {adata.n_vars:,} genes fetched  ({time.time()-t2:.0f}s)")

    t3 = time.time()
    print("  [4/5] Normalizing (cells already QC'd by fetch_metadata)...")
    adata = normalize(adata)
    print(f"       Done  ({time.time()-t3:.0f}s)")

    t4 = time.time()
    print(f"  [5/5] Selecting top {n_hvg} HVGs...")
    adata = select_hvg(adata, n_top_genes=n_hvg)
    print(f"       Done  ({time.time()-t4:.0f}s)  |  total: {time.time()-t0:.0f}s")

    return adata.var["feature_name"].tolist()


# ---------------------------------------------------------------------------
# PyTorch dataset + dataloader helpers
# ---------------------------------------------------------------------------

def resolve_hvg_var_joinids(census, organism_key: str, hvg_genes: list[str]) -> list[int]:
    """
    Return one sorted soma var joinid per gene in hvg_genes.

    A handful of gene symbols (e.g. pseudoautosomal-region genes like IL3RA,
    CSF2RA — annotated on both X and Y with distinct soma_joinids but the same
    feature_name) match more than one var row. Keep exactly one per name, so
    the fetched matrix width always equals len(hvg_genes) — the model's input
    dim is fixed to that count. Sorted first so the kept row is deterministic
    regardless of TileDB read order.
    """
    var_df = (
        census["census_data"][organism_key]["ms"]["RNA"]["var"]
        .read(value_filter=f"feature_name in {hvg_genes}")
        .concat()
        .to_pandas()
    )
    var_df = var_df.sort_values("soma_joinid").drop_duplicates(subset="feature_name", keep="first")
    return sorted(var_df["soma_joinid"].tolist())


def build_census_dataset(
    census,
    soma_joinids: list[int],
    hvg_genes: list[str],
    batch_size: int = 256,
    io_batch_size: int = 65_536,
    shuffle_chunk_size: int = 64,
    shuffle: bool = True,
    organism: str = "Homo sapiens",
) -> ExperimentDataset:
    """
    Build an ExperimentDataset for a donor-split cell population.

    Takes an already-open census handle so the caller controls context lifetime
    (keep it open for the duration of training — workers reopen via stored URI).
    """
    obs_joinids = sorted(soma_joinids)
    organism_key = organism.lower().replace(" ", "_")
    var_joinids = resolve_hvg_var_joinids(census, organism_key, hvg_genes)

    with census["census_data"][organism_key].axis_query(
        measurement_name="RNA",
        obs_query=tiledbsoma.AxisQuery(coords=(obs_joinids,)),
        var_query=tiledbsoma.AxisQuery(coords=(var_joinids,)),
    ) as query:
        return ExperimentDataset(
            query,
            layer_name="raw",
            batch_size=batch_size,
            io_batch_size=io_batch_size,
            shuffle=shuffle,
            shuffle_chunk_size=shuffle_chunk_size,
            obs_column_names=["soma_joinid", "cell_type"],
            use_eager_fetch=True,
        )


class CensusCollateFn:
    """
    Picklable collate callable for experiment_dataloader.

    Applies the same normalize_total(1e4) + log1p as preprocessing.normalize(),
    and maps cell_type strings to integer labels. Each item from ExperimentDataset
    is (X_ndarray, obs_df) where X has shape (batch_size, n_genes) and contains
    raw counts already restricted to QC-passed cells (see fetch_metadata) and the
    fixed HVG gene set.
    """

    def __init__(self, label2int: dict[str, int]):
        self.label2int = label2int

    def __call__(self, item):
        X_batch, obs_batch = item
        known = obs_batch["cell_type"].isin(self.label2int).values
        X = X_batch[known].astype(np.float32)
        counts_per_cell = X.sum(axis=1, keepdims=True)
        counts_per_cell[counts_per_cell == 0] = 1.0
        X = X / counts_per_cell * 1e4
        X = torch.from_numpy(np.log1p(X))
        cell_types = obs_batch["cell_type"].values[known]
        y = torch.tensor([self.label2int[ct] for ct in cell_types], dtype=torch.long)
        return X, y


# ---------------------------------------------------------------------------
# Download to local strategy.
# ---------------------------------------------------------------------------

def download_donor_h5ads(
    meta: pd.DataFrame,
    hvg_genes: list[str],
    output_dir: Path,
    organism: str = "Homo sapiens",
    census_version: str = CENSUS_VERSION,
) -> list[Path]:
    """
    Fetch cells donor-by-donor and write one h5ad file per donor to output_dir.

    Each donor is fetched and written independently, so memory use is bounded
    by a single donor's slice — never the full `meta` population. `meta` must
    already be QC'd (see fetch_metadata) and carry soma_joinid/donor_id/cell_type.
    All donor files share the same fixed HVG gene columns, in the same order
    (see resolve_hvg_var_joinids), so they stay directly concatenable/comparable
    downstream (e.g. anndata.experimental.AnnCollection or concat_on_disk).

    Idempotent / resumable: a donor whose "{donor_id}.h5ad" already exists is
    skipped, so an interrupted run can be picked back up by just rerunning with
    the same output_dir. Each file is written to a ".tmp" sibling first and
    atomically renamed into place on success — a crash mid-write leaves only
    a stray ".tmp" file, never a truncated "{donor_id}.h5ad" that a resume
    would mistake for already-downloaded.

    Files are named "{donor_id}.h5ad".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    organism_key = organism.lower().replace(" ", "_")
    written = []

    with cellxgene_census.open_soma(census_version=census_version, context=SOMA_CTX) as census:
        var_joinids = resolve_hvg_var_joinids(census, organism_key, hvg_genes)

        donor_groups = meta.groupby("donor_id", observed=True)
        pbar = tqdm(donor_groups, total=donor_groups.ngroups, desc="Downloading", unit="donor")
        for donor_id, donor_meta in pbar:
            path = output_dir / f"{donor_id}.h5ad"
            if path.exists():
                written.append(path)
                pbar.set_postfix(donor=donor_id, status="already downloaded")
                continue

            obs_joinids = sorted(donor_meta["soma_joinid"].tolist())

            with census["census_data"][organism_key].axis_query(
                measurement_name="RNA",
                obs_query=tiledbsoma.AxisQuery(coords=(obs_joinids,)),
                var_query=tiledbsoma.AxisQuery(coords=(var_joinids,)),
            ) as query:
                adata = query.to_anndata(
                    X_name="raw",
                    column_names={
                        "obs": ["soma_joinid", "cell_type", "donor_id"],
                        "var": ["feature_name"],
                    },
                )

            tmp_path = output_dir / f"{donor_id}.h5ad.tmp"
            adata.write_h5ad(tmp_path)
            tmp_path.rename(path)
            written.append(path)
            pbar.set_postfix(donor=donor_id, cells=f"{adata.n_obs:,}")

    return written