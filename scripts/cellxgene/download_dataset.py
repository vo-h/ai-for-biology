#!/usr/bin/env python3
"""
Download a QC'd, HVG-restricted Census slice to local disk as one h5ad per donor.

Each donor is fetched and written independently, so memory use is bounded by
a single donor's slice — never the full requested population. All donor files
share the same fixed HVG gene columns, in the same order, so they stay
directly concatenable/comparable downstream (e.g. anndata.experimental
.AnnCollection or concat_on_disk). See docs/cellxgene.md.

Example:
    python scripts/cellxgene/download_dataset.py \
        --tissues blood \
        --n-donors 10 \
        --hvg-cache results/hvg.json \
        --output-dir results/donors/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.cellxgene import compute_hvg_list, download_donor_h5ads, fetch_metadata


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tissues", nargs="+", default=["blood"])
    p.add_argument("--n-donors", type=int, default=10)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-sample-hvg", type=int, default=20_000,
                   help="Cells to sample for HVG computation")
    p.add_argument("--output-dir", type=Path, default=Path("results/donors"))
    p.add_argument("--census-version", default="2025-11-08")
    p.add_argument("--hvg-cache", type=Path, default=None,
                   help="Path to a saved HVG list JSON (skips recomputation)")
    p.add_argument("--start-after-donor", default=None,
                   help="Skip all donors sorting at or before this donor_id "
                        "(donors are visited in sorted order — see "
                        "download_donor_h5ads). Useful for resuming a "
                        "deliberately split/paginated download across "
                        "separate invocations; download_donor_h5ads already "
                        "skips donors already downloaded within a single run.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # HVG list — compute once and cache so reruns are fast
    hvg_cache = args.hvg_cache or args.output_dir / "hvg_list.json"
    if hvg_cache.exists():
        hvg_genes = json.loads(hvg_cache.read_text())
        print(f"Loaded {len(hvg_genes)} HVGs from {hvg_cache}")
    else:
        print(f"Computing HVG list from {args.n_sample_hvg:,} cells...")
        hvg_genes = compute_hvg_list(
            tissues=args.tissues,
            n_sample_cells=args.n_sample_hvg,
            n_hvg=args.n_hvg,
            census_version=args.census_version,
        )
        hvg_cache.write_text(json.dumps(hvg_genes))
        print(f"Saved {len(hvg_genes)} HVGs to {hvg_cache}")

    print("Fetching metadata (QC applied here — see fetch_metadata)...")
    meta = fetch_metadata(tissues=args.tissues, census_version=args.census_version)

    if args.start_after_donor is not None:
        # donor_id comes back as an unordered pandas Categorical (Census's
        # dictionary-encoded columns) — only equality works on that directly,
        # so compare as plain strings instead.
        # Filtered before the --n-donors cap below: capping first would pick
        # the top N donors by cell count irrespective of donor_id, then this
        # filter could easily exclude all of them even though later donors
        # exist — filter the candidate pool first, then cap what remains.
        meta = meta[meta["donor_id"].astype(str) > args.start_after_donor].copy()
        print(f"Skipping donors at or before {args.start_after_donor!r}")

    if meta["donor_id"].nunique() > args.n_donors:
        top = meta["donor_id"].value_counts().head(args.n_donors).index
        meta = meta[meta["donor_id"].isin(top)].copy()

    print(f"{len(meta):,} cells | {meta['donor_id'].nunique()} donors")

    print(f"Downloading to {args.output_dir}...")
    paths = download_donor_h5ads(
        meta,
        hvg_genes,
        output_dir=args.output_dir,
        census_version=args.census_version,
    )
    print(f"\nWrote {len(paths)} donor files to {args.output_dir}")


if __name__ == "__main__":
    main()
