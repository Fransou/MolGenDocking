#!/bin/bash
#SBATCH --job-name=orz_mol
#SBATCH --account=def-ibenayed
#SBATCH --time=72:00:00
#SBATCH --gpus=h100:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=molgendata

cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz

cd $WORKING_DIR
cp data/properties.csv $SLURM_TMPDIR

module load cuda

export DATA_PATH=$SLURM_TMPDIR/$DATASET
source $HOME/OpenRLHF/bin/activate
port=6379

wandb offline
export GPUS_PER_NODES=1
export PRETRAIN=$SCRATCH/ether0

ray start --head --node-ip-address 0.0.0.0


#export DEBUG_MODE=1
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"setup_commands": ["wandb offline"]}' \
   -- python3 -m openrlhf.cli.batch_inference \
   --micro_batch_size 16 \
   --pretrain $PRETRAIN \
   --max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --rollout_batch_size 16 \
   --dataset $DATA_PATH/eval_data/eval_prompts \
   --input_key prompt \
   --eval_task generate \
   --output_path $SCRATCH/MolGenOutput/generations.json

