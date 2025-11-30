#!/bin/bash
#SBATCH --job-name=reinvent-heterogenous
#SBATCH --account=def-ibenayed
#SBATCH --time=0-11:59:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --gpus=h100:1
#SBATCH --array=0-6:6

set -x
STRIDE=6

SLURM_SCRIPTS_DIR=$HOME/MolGenDocking/slurm/baselines

# Split nodes
NODES_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES_LIST[0]}
HEAD_NODE_IP=$(srun --nodes 1 --nodelist=$HEAD_NODE --ntasks=1 hostname --ip-address | head -n1)

# Launch head server
srun --overlap \
     --nodes 1 --nodelist=$HEAD_NODE --ntasks=1 \
     $SLURM_SCRIPTS_DIR/run_server.sh $HEAD_NODE_IP head &

export HEAD_NODE_PORT=34567
sleep 30

START_IDX=$SLURM_ARRAY_TASK_ID
END_IDX=$((START_IDX + STRIDE - 1))
for i in $(seq $START_IDX $END_IDX); do
    echo "Launching training for index $i"
    echo "SIGMA: $1"
    echo "N_BEAMS: $2"
    echo "TRAIN_ON_BEAMS: $3"
    echo "BATCH_SIZE: $4"
    srun --overlap $SLURM_SCRIPTS_DIR/reinvent/run_training.sh $HEAD_NODE_IP $i docking_only \
      $1 \
      $2 \
      $3 \
      $4
done

wait

echo "Training + reward server job finished."
