#!/bin/bash
# Run B2 and A0 variants on the ring dataset, 8 seeds each.
# This script waits for A2 runs to finish first (checks for the parent process).
# Usage: nohup bash scripts/run_b2_a0_seeds.sh > runs/b2_a0_seeds.log 2>&1 &
set -e

PYTHON="/home/neuro-voltron/.venv/bin/python"
SCRIPT="scripts/train_variant.py"
DATA="data/three_region_ring.npz"

cd /home/neuro-voltron

# Wait for A2 runs to finish (poll for the ring_seeds.sh process)
echo "Waiting for A2 runs to finish..."
while pgrep -f "run_ring_seeds.sh" > /dev/null 2>&1; do
    sleep 60
done
echo "A2 runs complete. Starting B2 and A0 at $(date)"
echo ""

# === B2 variant: 8 seeds ===
CONFIG_B2="configs/B2_v3.json"
for SEED in 0 1 2 3 4 5 6 7; do
    WORKDIR="runs/B2_v3_ring_seed${SEED}"
    echo "============================================"
    echo "B2 seed=${SEED} -> ${WORKDIR}"
    echo "Time: $(date)"
    echo "============================================"

    $PYTHON $SCRIPT \
        --config-json $CONFIG_B2 \
        --data $DATA \
        --workdir $WORKDIR \
        --seed $SEED

    echo "Finished B2 seed=${SEED} at $(date)"
    echo ""
done

echo "============================================"
echo "All B2 seeds complete at $(date)"
echo "============================================"
echo ""

# === A0 variant: 8 seeds ===
CONFIG_A0="configs/A0_v3.json"
for SEED in 0 1 2 3 4 5 6 7; do
    WORKDIR="runs/A0_v3_ring_seed${SEED}"
    echo "============================================"
    echo "A0 seed=${SEED} -> ${WORKDIR}"
    echo "Time: $(date)"
    echo "============================================"

    $PYTHON $SCRIPT \
        --config-json $CONFIG_A0 \
        --data $DATA \
        --workdir $WORKDIR \
        --seed $SEED

    echo "Finished A0 seed=${SEED} at $(date)"
    echo ""
done

echo "============================================"
echo "All B2 + A0 seeds complete at $(date)"
echo "============================================"
