#!/usr/bin/env bash
set -euo pipefail

OUT_DIR='output/vae3d'
CFG_FILE='config/vae_kl_ft2_fs8_z16.yaml'
RESUME_DIR="${OUT_DIR}"

mkdir -p "${OUT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 main_vae.py \
  --batch_size 2 \
  --epochs 100 \
  --lr 4e-4 \
  --weight_decay 0.01 \
  --disc_loss_scale 0.1 \
  --save_last_freq 1 \
  --persistent_workers \
  --no_amp \
  --resume "${RESUME_DIR}" \
  --output_dir "${OUT_DIR}" \
  --log_dir "${OUT_DIR}" \
  --cfg "${CFG_FILE}"
