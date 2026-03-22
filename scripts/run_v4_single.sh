#!/bin/bash
set -e
cd /home/neuro-voltron
echo "Starting A2_v4 seed=0 at $(date)"
.venv/bin/python scripts/train_variant.py \
    --config-json configs/A2_v4.json \
    --data data/three_region_ring.npz \
    --workdir runs/A2_v4_ring_seed0 \
    --seed 0
echo "Finished at $(date)"
