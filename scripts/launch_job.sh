#!/usr/bin/env bash
# Usage: ./scripts/launch_job.sh <config.json> <data.npz> <workdir> [seed]
# Launches a single training job with duplicate-process protection.
# Logs go to <workdir>/run.log (clean, no tqdm noise).

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:?Usage: launch_job.sh <config> <data> <workdir> [seed]}"
DATA="${2:?Missing data path}"
WORKDIR="${3:?Missing workdir}"
SEED="${4:-0}"

# Duplicate protection: check for existing process with same workdir
if pgrep -f "train_variant.py.*--workdir ${WORKDIR}" > /dev/null 2>&1; then
    echo "ERROR: A training process is already running for ${WORKDIR}"
    echo "  PIDs: $(pgrep -f "train_variant.py.*--workdir ${WORKDIR}")"
    echo "  Kill it first if you want to restart."
    exit 1
fi

# Create workdir
mkdir -p "${WORKDIR}"

LOG="${WORKDIR}/run.log"

echo "Launching: config=${CONFIG} data=${DATA} workdir=${WORKDIR} seed=${SEED}"
echo "Log: ${LOG}"

nohup .venv/bin/python scripts/train_variant.py \
    --config-json "${CONFIG}" \
    --data "${DATA}" \
    --workdir "${WORKDIR}" \
    --seed "${SEED}" \
    > "${LOG}" 2>&1 &

PID=$!
echo "${PID}" > "${WORKDIR}/pid"
echo "Started PID ${PID}"
echo "Monitor with: tail -f ${LOG}"
echo "Or clean log: tail -f ${WORKDIR}/train.log"
