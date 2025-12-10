#!/bin/bash
#SBATCH --job-name=reinvent-rpo
#SBATCH --account=def-ibenayed
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=100G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=h100:4
#SBATCH --nodes=1
#SBATCH --array=0-125:4
STRIDE=4

SLURM_SCRIPTS_DIR=$HOME/MolGenDocking/slurm/baselines
# Split nodes
NODES_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES_LIST[0]}
HEAD_NODE_IP=$(srun --nodes 1 --nodelist=$HEAD_NODE --ntasks=1 hostname --ip-address | head -n1)
echo "Head node IP: $HEAD_NODE_IP"


# Launch head server
bash $SLURM_SCRIPTS_DIR/run_server.sh $HEAD_NODE_IP head &

export HEAD_NODE_PORT=34567
sleep 45

START_IDX=$SLURM_ARRAY_TASK_ID
END_IDX=$((START_IDX + STRIDE - 1))
GPU_ID=0
TRAIN_PIDS=()

for i in $(seq $START_IDX $END_IDX); do
    echo "Launching training for index $i on GPU ${GPU_ID}"
    echo "SIGMA: $1"
    echo "N_BEAMS: $2"
    echo "TRAIN_ON_BEAMS: $3"
    echo "BATCH_SIZE: $4"
    echo "LR: $5"
    echo "REF: $6"
    CUDA_VISIBLE_DEVICES=$GPU_ID bash $SLURM_SCRIPTS_DIR/reinvent_rpo/run_training.sh \
      $HEAD_NODE_IP $i all \
      $1 $2 $3 $4 $5 $6 &
    TRAIN_PIDS+=($!)
    GPU_ID=$((GPU_ID + 1))
done

wait "${TRAIN_PIDS[@]}"
echo "Training + reward server job finished."
