#!/usr/bin/env bash
# Runs Phase 3's 3 training runs sequentially, unattended.
# Waits for data/phase2/prepare_full.py's output before starting, since all
# 3 runs need the full-corpus tokenized bins.
#
# Usage: nohup ./scripts/run_phase3.sh > logs/phase3_queue.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

DATA_DIR="data/phase2/full"
BATCH_SIZE=8
BLOCK_SIZE=1024
TOKENS_PER_ITER=$((BATCH_SIZE * BLOCK_SIZE))

mkdir -p logs

echo "[queue] waiting for full-corpus tokenization ($DATA_DIR/train.bin, val.bin)..."
while [ ! -f "$DATA_DIR/train.bin" ] || [ ! -f "$DATA_DIR/val.bin" ]; do
  sleep 30
done
echo "[queue] tokenized corpus ready, starting runs"

run() {
  local name=$1 n_layer=$2 n_head=$3 n_embd=$4 tokens=$5
  local max_iters=$((tokens / TOKENS_PER_ITER))
  local out_dir="checkpoints/phase3/${name}/"
  local ckpt="${out_dir}best.pt"

  if [ -f "${out_dir}final.pt" ]; then
    echo "[queue] === $name already complete, skipping ==="
    return
  fi

  local attempt=1 max_attempts=5
  while [ $attempt -le $max_attempts ]; do
    local resume_arg=""
    if [ -f "$ckpt" ]; then
      resume_arg="--resume_from $ckpt"
      echo "[queue] === $name attempt $attempt: resuming from $ckpt ==="
    else
      echo "[queue] === $name attempt $attempt: n_layer=$n_layer n_head=$n_head n_embd=$n_embd tokens=$tokens max_iters=$max_iters ==="
    fi

    if python3 src/phase3/train.py \
      --n_layer "$n_layer" --n_head "$n_head" --n_embd "$n_embd" \
      --max_iters "$max_iters" \
      --data_dir "$DATA_DIR/" \
      --out_dir "$out_dir" \
      --wandb_run_name "phase3-$name" \
      --no_compile \
      $resume_arg \
      >> "logs/phase3_${name}.log" 2>&1; then
      echo "[queue] === $name done ==="
      return
    fi

    echo "[queue] === $name attempt $attempt crashed, retrying from checkpoint in 30s ==="
    sleep 30
    attempt=$((attempt + 1))
  done

  echo "[queue] === $name FAILED after $max_attempts attempts, aborting queue ==="
  exit 1
}

COOLDOWN_SECONDS=600

cooldown() {
  echo "[queue] cooling down for $((COOLDOWN_SECONDS / 60)) min before next run..."
  sleep "$COOLDOWN_SECONDS"
}

run "75m_500m" 8 8 576 500000000
cooldown
run "75m_2b"   8 8 576 2000000000
cooldown
run "150m_500m" 12 12 768 500000000

echo "[queue] all 3 runs complete"
