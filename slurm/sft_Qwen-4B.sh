#!/bin/bash
#SBATCH --job-name=sft-Qwen-4B
#SBATCH --account=def-ibenayed
#SBATCH --time=12:00:00
#SBATCH --gpus=h100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
cd $WORKING_DIR

module load cuda
source $HOME/OpenRLHF/bin/activate

wandb offline
export GPUS_PER_NODES=1
export PRETRAIN=$SCRATCH/Qwen/Qwen3-4B

#export DEBUG_MODE=1
export HF_HUB_OFFLINE=1
accelerate launch mol_gen_docking/sft/sft.py \
    --output_dir $SCRATCH/Qwen/sft_Qwen-4B \
    --batch_size 6 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --model_name $PRETRAIN \
    --local-files-only \
    --dataset SMolInstruct