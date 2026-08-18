# Multi-Node Training: Single-GPU vs. 2-GPU DDP Speedup

**Question:** does splitting one epoch of RxRx1 training across 2 single-GPU nodes actually halve wall-clock time, once real DDP communication overhead — not just raw GPU compute — is accounted for?

## Background

The naive expectation for data-parallel training is "2 GPUs → 2x throughput." That ignores gradient synchronization cost: every DDP step ends in an `all_reduce` across ranks, and unlike a single multi-GPU box (NVLink, shared PCIe switch), two separate single-GPU instances sync gradients over the ordinary VPC network. A `torch.profiler` capture from an earlier 2-node RxRx1 run (see `results/node1/rxrx1/traces/`) already found `nccl:all_reduce` consuming ~25% of every training step, with the GPU otherwise 100% saturated and the dataloader a complete non-issue (<0.1% of step time). If that number holds up at the full-epoch, aggregate level, the honest expectation is real-but-sub-2x speedup, not a clean doubling — that's the actual thing this project measures.

Hardware is 2x `g4dn.xlarge` (1x NVIDIA T4 each) rather than one multi-GPU instance because no multi-GPU NVIDIA instance type fits under this AWS account's 32-vCPU on-demand quota — see `terraform/` and the commit history for that investigation. That constraint is itself part of what makes this comparison worth doing: this is the realistic "no budget for a real multi-GPU box" scenario, not an idealized one.

## Method

- **Data:** RxRx1 (`rxrx1-us-central1`), full `train` split, all 4 cell lines, both imaging sites — identical dataset for both configurations.
- **Model:** `RxRx1ResNet18` (`src/models/resnet.py`), identical architecture/hyperparameters both runs.
- **Configurations:**
  - 1x `g4dn.xlarge` (single GPU, no DDP communication)
  - 2x `g4dn.xlarge` (DDP, `--rdzv_backend=static` — see `scripts/run_remote.sh`)
- **1 epoch only** (`--n-epochs 1`): this project measures wall-clock throughput, not model quality — see [Split Strategies](split-strategies.md) for the accuracy-focused work in this repo. 1108-way classification accuracy after 1 epoch is expected to be near-random and isn't the point.
- **Infra:** `scripts/run_remote.sh` (provisioning, detached launch, log streaming, results-pulling — see its own `--help`), `torch.profiler` wiring in `src/training/rxrx1.py` (`--profile`, per-rank traces land in `results/nodeN/rxrx1/traces/`).
- **Metrics:** `timing.avg_images_per_s` and `timing.total_wall_time_s` from each run's `results/nodeN/rxrx1/run_metadata.json`, plus a `--profile` trace per run to break the aggregate wall-clock gap down into actual compute vs. `nccl:all_reduce` time rather than just inferring the split.

[PLACEHOLDER — exact run names, instance IPs, and timestamps once actually run]

## Results

[PLACEHOLDER — no runs yet]

| Config | Wall time (1 epoch) | Images/sec (aggregate) | Speedup vs. single-node |
|---|---|---|---|
| 1x g4dn.xlarge | TBD | TBD | 1.00x (baseline) |
| 2x g4dn.xlarge (DDP) | TBD | TBD | TBD |

[PLACEHOLDER — profiler-derived breakdown of where the 2-node run's time goes (compute vs. `nccl:all_reduce`), compared against the single-node run's breakdown (compute only, no communication)]

## Future work

- [PLACEHOLDER]

## Reproduce

```bash
# Single-node baseline: 1x g4dn.xlarge, 1 GPU
scripts/run_remote.sh --instance-type g4dn.xlarge --name rxrx1-single -- \
    torchrun --nproc_per_node=1 scripts/rxrx1/train_resnet.py --n-epochs 1 --batch-size 32 --profile

# Two-node: 2x g4dn.xlarge, 1 GPU each, DDP
scripts/run_remote.sh --instance-type g4dn.xlarge --nodes 2 --name rxrx1-multi -- \
    torchrun --nproc_per_node=1 scripts/rxrx1/train_resnet.py --n-epochs 1 --batch-size 32 --profile
```

Compare `timing.total_wall_time_s` / `timing.avg_images_per_s` in each run's `results/node0/rxrx1/run_metadata.json` once both finish. `--profile` writes a per-rank Chrome trace to `results/nodeN/rxrx1/traces/` (open at `chrome://tracing` or `ui.perfetto.dev`) — the single-node trace has no `nccl:all_reduce` at all (nothing to sync with), so diffing it against a 2-node rank's trace isolates exactly how much of each step is communication vs. compute, rather than just inferring it from the aggregate wall-clock gap.
