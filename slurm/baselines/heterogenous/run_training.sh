#!/bin/bash

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=molgendata
export NCCL_ASYNC_ERROR_HANDLING=1

cp $SCRATCH/MolGenData/$DATASET_prompts.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz
cd $WORKING_DIR

export DATA_PATH=$SLURM_TMPDIR/$DATASET
source $HOME/OpenRLHF/bin/activate
wandb offline
HF_HUB_OFFLINE=1 python mol_gen_docking/baselines/reinvent/rl_opt.py \
  --model_name reinvent_10M_prior \
  --dataset $DATA_PATH/eval_data/eval_prompts \
  --datasets-path $DATA_PATH \
  --batch_size 64 \
  --sigma 1 \
  --num_train_epochs 500 \
  --generation_config '{"num_beams": 2}' \
  --train_on_beams 0 \
  --id_obj $2 \
  --remote_rm_url http://$1:5001
