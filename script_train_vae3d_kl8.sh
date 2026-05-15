#!/usr/bin/env bash
set -euo pipefail

OUT_DIR='output/vae3d'
CFG_FILE='config/vae_kl_ft2_fs8_z16.yaml'

mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 main_vae.py \
  --batch_size 10 \
  --epochs 100 \
  --lr 8e-4 \
  --weight_decay 0.01 \
  --disc_loss_scale 0.1 \
  --limit_num 10 \
  --save_last_freq 10 \
  --eval_freq 10 \
  --online_eval \
  --persistent_workers \
  --output_dir "${OUT_DIR}" \
  --log_dir "${OUT_DIR}" \
  --cfg "${CFG_FILE}"
