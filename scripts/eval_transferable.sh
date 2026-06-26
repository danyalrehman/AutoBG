#!/bin/bash
# =============================================================================
# Autoregressive Boltzmann Generators (ArBG) -- transferable evaluation (Robin)
# =============================================================================
# Samples from Robin and evaluates zero-shot generalization on unseen peptides
# (E-W2 / T-W2 / TICA-W2 via SNIS reweighting).
#
# Checkpoint: set CKPT to a local .ckpt, or leave it unset to auto-download the
# released Robin checkpoint from Hugging Face (danyalrehman17/robin-transferable).
#
# Evaluates all 90 held-out test peptides (30 each of length 2, 4, and 8).
#
# Usage:
#   sbatch scripts/eval_transferable.sh                                     # Robin from HF
#   CKPT=/path/to/robin.ckpt sbatch scripts/eval_transferable.sh   # local checkpoint
# =============================================================================

#SBATCH -J arbg_transferable_eval
#SBATCH -o watch_folder/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH -t 72:00:00
#SBATCH --get-user-env
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@500

RUN_NAME="robin_transferable_up_to_8aa"

# Checkpoint: use a local CKPT if given, otherwise download the released Robin
# checkpoint (a full Lightning .ckpt) from Hugging Face.
CKPT="${CKPT:-}"
if [ -z "${CKPT}" ]; then
  echo "No CKPT set; downloading released Robin checkpoint from Hugging Face..."
  CKPT="$(hf download danyalrehman17/robin-transferable robin.ckpt)"
fi

# All 90 held-out test peptides (30 each of length 2, 4, and 8) from the paper.
SEQS_2AA="AA,CE,CL,DG,DI,FK,HL,HM,IK,IM,KG,LE,MQ,NA,NC,PG,PY,QR,RL,RT,SS,TD,VF,VS,WA,WH,WQ,WS,YC,YQ"
SEQS_4AA="ARIP,CCVH,CIPQ,DEMT,DMTL,EHQW,FESD,FYYY,GCDE,GDTI,GGRS,HEAV,HQVS,HYGW,ITYL,KKAP,KLLR,KRWN,NCFG,NEVI,PQIF,QAKR,QWNL,RLMM,SHKS,SVND,TAPF,TMWC,VPFY,WNMA"
SEQS_8AA="ANKSMIEA,CGSWHKQR,CLCCGQWN,DDRDTEQT,DGVAHALS,EKYYWMQT,FWRVDHDM,GNDLVTVI,HWHSLICK,IDHRQLKW,IFGWVYTG,ISKCKNGE,KRRGFFLE,MAPQTIAT,MRDPVLFA,MWNSTEMI,MYGRNCYM,NHQYGSDP,NKEKFFQH,NPCLCYML,PGESTAES,PLFHVMYV,PPWRECNN,PYIRNCVE,SPHKMRLC,SQQKVAFE,VWIPVIDT,WDLIQFRQ,WTYAFAHS,YFPHAGYT"
TEST_SEQUENCES="${SEQS_2AA},${SEQS_4AA},${SEQS_8AA}"

python src/eval.py \
  experiment=training/transferable/autoregressive_up_to_8aa \
  data=transferable/up_to_8aa \
  "data.test_sequences=[${TEST_SEQUENCES}]" \
  data.num_workers=2 \
  model.sampling_config.batch_size=1000 \
  model.sampling_config.num_test_proposal_samples=200000 \
  trainer=gpu \
  ckpt_path="${CKPT}" \
  logger.wandb.id="${RUN_NAME}_eval" \
  hydra.run.dir='${paths.log_dir}/${task_name}/runs/'"${RUN_NAME}"_eval \
  tags=[arbg,robin,transferable,eval,up_to_8aa]
