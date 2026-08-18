#!/usr/bin/env bash
set -e
cd "/home/ubuntu/ai-for-biology"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
exec torchrun --nnodes=2 --node_rank=1 --rdzv_id=rxrx1-multi --rdzv_backend=static --rdzv_endpoint=172.31.40.106:29500 --rdzv_conf=timeout=1800 --nproc_per_node=1 scripts/rxrx1/train_resnet.py --n-epochs 1 --batch-size 32 --profile 
