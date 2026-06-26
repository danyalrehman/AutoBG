#!/bin/bash
# =============================================================================
# Autoregressive Boltzmann Generators (ArBG) -- single-system training / eval
# =============================================================================
# Trains (or evaluates) an ArBG on a single peptide. This example uses
# tri-alanine (AAA). Set TRAIN=false to skip training and instead sample +
# evaluate a trained checkpoint (E-W2 / T-W2 / TICA-W2 via SNIS reweighting).
#
# Usage:
#   sbatch scripts/train_single_system.sh                       # train
#   TRAIN=false CKPT=/path/to/last.ckpt sbatch scripts/train_single_system.sh   # evaluate
# =============================================================================

#SBATCH -J arbg_single_system
#SBATCH -o watch_folder/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH --get-user-env
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@500

# --- toggle: true = train, false = evaluate an existing checkpoint ----------
TRAIN="${TRAIN:-true}"
RUN_NAME="arbg_AAA_single_system"

# When evaluating (TRAIN=false), CKPT must point at a trained checkpoint.
CKPT="${CKPT:-}"
CKPT_ARG=""
if [ "${TRAIN}" = "false" ]; then
  if [ -z "${CKPT}" ]; then
    echo "ERROR: TRAIN=false requires CKPT=/path/to/checkpoint.ckpt"; exit 1
  fi
  CKPT_ARG="ckpt_path=${CKPT}"
fi

srun uv run python src/train.py \
  experiment=training/single_system/autoregressive_AAA \
  model/optimizer=muon \
  trainer=ddp \
  trainer.max_epochs=2000 \
  train="${TRAIN}" \
  ${CKPT_ARG} \
  seed=0 \
  logger.wandb.id="${RUN_NAME}" \
  hydra.run.dir='${paths.log_dir}/${task_name}/runs/'"${RUN_NAME}" \
  tags=[arbg,single_system,AAA]
