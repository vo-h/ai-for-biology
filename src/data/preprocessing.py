"""
QC, normalization, and feature selection for scRNA-seq data.

run_qc is not used by the training/eval pipeline — src.data.census.fetch_metadata
applies equivalent QC using Census's precomputed per-cell stats instead (see
docs/census.md for why). normalize + select_hvg are used once, upfront, via
compute_hvg_list; CensusCollateFn re-implements normalize's two steps as
vectorized numpy for per-batch use. There is no PCA step.
"""

from __future__ import annotations

import anndata as ad
import scanpy as sc


QC_PARAMS = {
    "min_genes": 200,       # below → likely empty droplet
    "max_genes": 6000,      # above → likely doublet
    "max_pct_mito": 20.0,   # above → likely damaged cell
    "min_counts": 500,
}

N_HVG = 2000


def run_qc(adata: ad.AnnData, params: dict | None = None) -> ad.AnnData:
    """Filter low-quality cells and return the filtered AnnData."""
    p = params or QC_PARAMS
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    mask = (
        (adata.obs["n_genes_by_counts"] >= p["min_genes"])
        & (adata.obs["n_genes_by_counts"] <= p["max_genes"])
        & (adata.obs["pct_counts_mt"] <= p["max_pct_mito"])
        & (adata.obs["total_counts"] >= p["min_counts"])
    )
    return adata[mask].copy()


def normalize(adata: ad.AnnData) -> ad.AnnData:
    """
    Total-count normalize to 10k counts then log1p.
    Raw counts saved to adata.layers["counts"] for HVG selection.
    """
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def select_hvg(adata: ad.AnnData, n_top_genes: int = N_HVG) -> ad.AnnData:
    """Select highly variable genes using seurat_v3 on raw counts."""
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        layer="counts",
    )
    return adata[:, adata.var["highly_variable"]].copy()


def get_label_encoder(labels: list[str]) -> tuple[dict, dict]:
    """Return (label→int, int→label) dicts. Sorted alphabetically for determinism."""
    unique = sorted(set(labels))
    label2int = {l: i for i, l in enumerate(unique)}
    int2label = {i: l for l, i in label2int.items()}
    return label2int, int2label
