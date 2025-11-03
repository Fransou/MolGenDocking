#!/bin/bash
#SBATCH --job-name=reinvent-heterogenous
#SBATCH --account=def-ibenayed
#SBATCH --time=0-00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --gpus=h100:1
#SBATCH --array=0-3

set -x
i=$SLURM_ARRAY_TASK_ID

SLURM_SCRIPTS_DIR=$HOME/MolGenDocking/slurm/baselines/heterogenous
# Split nodes
NODES_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES_LIST[0]}
WORKER_NODES=${NODES_LIST[@]:1}
echo "Head node: $HEAD_NODE"
echo "Worker nodes: $WORKER_NODES"

# Get IP of head node
HEAD_NODE_IP=$(srun --nodes 1 --nodelist=$HEAD_NODE --ntasks=1 hostname --ip-address | head -n1)

# Launch head server
srun  --overlap --nodes 1 --nodelist=$HEAD_NODE --ntasks=1 $SLURM_SCRIPTS_DIR/run_server.sh $HEAD_NODE_IP head &

export HEAD_NODE_PORT=34567
sleep 15
srun --overlap $SLURM_SCRIPTS_DIR/run_training.sh $HEAD_NODE_IP $i

wait

echo "Training + reward server job finished."
