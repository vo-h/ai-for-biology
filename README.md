# AI for Biology

A collection of applied ML projects on biological data — starting with the [CZ CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/) (~162M human single cells). Shared infrastructure (Census streaming + QC, local dataset caching, training loop) lives in `src/`; each project below is a self-contained question asked of that infrastructure.

## Projects

| Project | Question | Status |
|---|---|---|
| [Split Strategies](projects/split-strategies.md) | Does cross-donor evaluation actually differ from a random split? | Active |
| [Multi-Node Training](projects/multi-node-training.md) | Does splitting one epoch of RxRx1 training across 2 single-GPU nodes actually halve wall-clock time, once real DDP communication overhead is accounted for? | Active |

## Docs & code

- **`docs/`** — quick overview of the shared codebase. Start at [docs/README.md](docs/README.md).
- **`projects/`** — one file per project: motivation, method, results, future work.

## Setup

```bash
pip install -r requirements.txt
```

Nothing is downloaded by default — Census access streams directly from S3 (LTS release `2025-11-08`, pinned for reproducibility). See a project's doc for how to run it.
