#!/bin/bash
# =============================================================================
# Autoregressive Boltzmann Generators (ArBG) -- transferable training (Robin)
# =============================================================================
# Trains Robin, the 132M-parameter transferable ArBG, across the ManyPeptidesMD
# dataset (peptides up to 8 residues). Conditioning is on atom type, residue
# type, residue position, and sequence length.
#
# Usage:
#   sbatch scripts/train_transferable.sh
# =============================================================================

#SBATCH -J arbg_transferable
#SBATCH -o watch_folder/%x_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH -c 4
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH --get-user-env
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@500

RUN_NAME="robin_transferable_up_to_8aa"

srun uv run python src/train.py \
  experiment=training/transferable/autoregressive_up_to_8aa \
  model/optimizer=muon \
  data.batch_size=448 \
  trainer=ddp \
  trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
  trainer.max_epochs=1000 \
  trainer.check_val_every_n_epoch=5000 \
  callbacks=default_with_last_only \
  callbacks.model_checkpoint_time.every_n_epochs=25 \
  logger.wandb.id="${RUN_NAME}" \
  hydra.run.dir='${paths.log_dir}/${task_name}/runs/'"${RUN_NAME}" \
  tags=[arbg,robin,transferable,up_to_8aa]
