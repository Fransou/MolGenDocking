#!/bin/bash
#SBATCH --job-name=batch_inference_molgen
#SBATCH --account=def-ibenayed
#SBATCH --time=06:00:00
#SBATCH --gpus=h100:4
#SBATCH --mem=248G
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=0-2

export WORKING_DIR=$HOME/MolGenDocking

source $HOME/.bashrc
source $HOME/OpenRLHF/bin/activate

export DATASET=molgendata
cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz
cd $WORKING_DIR
cp data/properties.csv $SLURM_TMPDIR
export DATA_PATH=$SLURM_TMPDIR/$DATASET

export PRETRAIN=$1

ray start --head --node-ip-address 0.0.0.0

export docking_oracle=autodock_gpu
export scorer_exhaustiveness=4

if [[ -n "$5" ]]; then
    TP_SIZE=$5
else
    TP_SIZE=4
fi
echo "TP size $TP_SIZE"
#export DEBUG_MODE=1
ray job submit \
   --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"setup_commands": ["wandb offline"]}' \
   -- python3 -m openrlhf.cli.batch_inference \
   --pretrain $PRETRAIN \
   --max_len 4096 \
   --max_new_tokens 32768 \
   --zero_stage 3 \
   --bf16 \
   --best_of_n 128 \
   --rollout_batch_size 100 \
   --iter $SLURM_ARRAY_TASK_ID \
   --apply_chat_template \
   --dataset $DATA_PATH/$3 \
   --input_key messages \
   --label_key meta \
   --eval_task generate_vllm \
   --output_path $2_$SLURM_ARRAY_TASK_ID \
   --top_p 0.95 \
   --temperature $4 \
   --system_prompt system_prompts/vanilla.json \
   --tp_size $TP_SIZE

python -m mol_gen_docking.score_completions \
    --input_file $2_$SLURM_ARRAY_TASK_ID.jsonl \
    --batch_size 1024
