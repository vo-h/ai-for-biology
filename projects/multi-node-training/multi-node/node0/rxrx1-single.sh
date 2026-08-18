#!/usr/bin/env bash
set -e
cd "/home/ubuntu/ai-for-biology"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
exec torchrun --standalone --nproc_per_node=1 scripts/rxrx1/train_resnet.py --n-epochs 1 --batch-size 32 --profile 
