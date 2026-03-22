#!/bin/bash
# Run A2_v4 training on the ring dataset with 8 different random seeds.
# Cyclical beta annealing + SGDR + enhanced monitoring.
# Usage: nohup bash scripts/run_v4_seeds.sh > runs/v4_seeds.log 2>&1 &
set -e

PYTHON="/home/neuro-voltron/.venv/bin/python"
SCRIPT="scripts/train_variant.py"
CONFIG="configs/A2_v4.json"
DATA="data/three_region_ring.npz"

cd /home/neuro-voltron

for SEED in 0 1 2 3 4 5 6 7; do
    WORKDIR="runs/A2_v4_ring_seed${SEED}"
    echo "============================================"
    echo "A2_v4 seed=${SEED} -> ${WORKDIR}"
    echo "Time: $(date)"
    echo "============================================"

    $PYTHON $SCRIPT \
        --config-json $CONFIG \
        --data $DATA \
        --workdir $WORKDIR \
        --seed $SEED

    echo ""
    echo "Finished seed=${SEED} at $(date)"
    echo ""
done

echo "============================================"
echo "All 8 A2_v4 seeds complete at $(date)"
echo "============================================"
