# Multi-Node Training: Does Out-of-the-Box DDP Actually Deliver?

**Question:** run RxRx1 on 1 GPU vs. 2 GPUs (2 nodes) with plain `torchrun`/DDP, no tuning. Does it just work and roughly speed things up, or is it one of those HPC setups that promises speedup but needs real tuning to get it?

## Background

Data-parallel training across nodes adds a gradient-sync cost (`all_reduce`) that a single GPU doesn't have, so 2 GPUs won't automatically mean 2x. This is a quick check of how close out-of-the-box DDP gets, not a rigorous scaling study.

Using 2x `g4dn.xlarge` (1 GPU each) instead of one multi-GPU box because no multi-GPU instance type fits this AWS account's 32-vCPU quota — see `terraform/`.

## Method

- RxRx1, full `train` split, all cell lines/sites, same model/hyperparameters both runs, `--batch-size 32`, 1 epoch (`--n-epochs 1` — speed check, not an accuracy run).
- Configs: 1x `g4dn.xlarge` vs. 2x `g4dn.xlarge` DDP (`--rdzv_backend=static`).
- `scripts/run_remote.sh` for launch/results, `torch.profiler` (`--profile`) for a quick look at where time goes.

Run names `rxrx1-single` / `rxrx1-multi`, 2026-08-17. Node 0 (`ip-172-31-40-106`) reused for both runs; node 1 was `ip-172-31-25-5`.

## DDP setup

`torchrun` sets `RANK`/`LOCAL_RANK`/`WORLD_SIZE` env vars per process; the script just reads them (`src/training/rxrx1.py`):

```python
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
torch.cuda.set_device(local_rank)

train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed)
train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=cfg.batch_size, ...)

model = RxRx1ResNet18(...).to(device)
ddp_model = DDP(model, device_ids=[local_rank])
```

`DistributedSampler` is what actually splits the data — each rank's sampler only yields indices `i where i % world_size == rank` (shuffled first, same `seed` on every rank so the pre-shuffle order matches). No manual slicing needed. It also pads the dataset up to a multiple of `world_size` so every rank gets the same number of batches — otherwise a rank that runs out early leaves the others hanging on an `all_reduce` that never comes, since DDP requires every rank to call `backward()` the same number of times per epoch.

## Results

| Config | Wall time | Images/sec | Speedup |
|---|---|---|---|
| 1x g4dn.xlarge | 7397s (123 min) | 9.9 | 1.00x |
| 2x g4dn.xlarge (DDP) | 4199s (70 min) | 17.4 | **1.76x** |

So: yes, it just worked — no tuning, default `torchrun`/DDP flags, and a real 1.76x out of 2 GPUs (88% efficiency). Val accuracy was near-random for both (1 epoch, 1108 classes) — expected, not the point.

One thing the `--profile` traces got wrong at a glance: the 3-step capture window showed NCCL's all-reduce almost fully overlapped with backward-pass compute (compute stream ~99.7% busy either way), suggesting only ~5% overhead. But the actual full-epoch numbers work out to ~14% per-batch overhead (`total_train_time_s` / batch count) — that's what actually accounts for 1.76x instead of 2x. The short profiler snapshot just isn't long enough to catch it; whatever adds the other ~9 points (network/GCS variance under sustained load, probably) doesn't show up in 3 batches at the start of an epoch.

## Problems along the way

DDP itself needed zero tuning once launched right — getting two separate EC2 instances to actually rendezvous reliably was most of the real work:

- **Rendezvous timeouts looked like a network problem — they weren't.** Spent a while chasing security groups/NACLs/conntrack before finding the real cause: `read_timeout` (60s default, initial TCPStore connection) and `join_timeout` (600s default, waiting for all ranks) were both too short next to the launch script's own retry overhead. Fix: bump both via `--rdzv_conf=join_timeout=1800,read_timeout=1800`.
- **Master's clock starts before workers even try to connect.** The rendezvous host opens its listening socket and starts waiting the moment it launches — if it launches first, its timeout window is already ticking while workers are still syncing code/installing deps. Fix: launch workers first, master last (`scripts/run_remote.sh` reverses launch order).
- **`--node_rank` silently ignored.** `--rdzv_backend=c10d` assigns global rank by *join order*, not the `--node_rank` you pass — so "node 0" isn't necessarily rank 0, and whichever rank happens to be main (the one that actually prints/writes files) becomes unpredictable. Fix: `--rdzv_backend=static`, which does honor `--node_rank`.
- **DataLoader workers segfaulted on the first batch.** Only showed up once rendezvous succeeded and training actually started — CUDA gets initialized in the parent process before DataLoader workers fork, and Linux's default fork-based worker start corrupts CUDA state in the child. Fix: `multiprocessing_context="spawn"`.

## Future work

- Profile a longer window (or `repeat>1` across the epoch) to see where the extra ~9% actually comes from.
- Try a real multi-GPU box if the vCPU quota ever allows one, as a same-node comparison point.
- Run more epochs for an actual accuracy number, now that speed's answered.

## Reproduce

```bash
# Single-node baseline: 1x g4dn.xlarge, 1 GPU
scripts/run_remote.sh --instance-type g4dn.xlarge --name rxrx1-single -- \
    torchrun --nproc_per_node=1 scripts/rxrx1/train_resnet.py --n-epochs 1 --batch-size 32 --profile

# Two-node: 2x g4dn.xlarge, 1 GPU each, DDP
scripts/run_remote.sh --instance-type g4dn.xlarge --nodes 2 --name rxrx1-multi -- \
    torchrun --nproc_per_node=1 scripts/rxrx1/train_resnet.py --n-epochs 1 --batch-size 32 --profile
```

Compare `timing.total_wall_time_s` / `timing.avg_images_per_s` in each run's `results/node0/rxrx1/run_metadata.json`. `--profile` traces land in `results/nodeN/rxrx1/traces/` — open at `chrome://tracing` or `ui.perfetto.dev`.
