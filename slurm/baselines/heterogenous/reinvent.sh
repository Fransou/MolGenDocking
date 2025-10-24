#!/bin/bash
#SBATCH --job-name=reinvent-heterogenous
#SBATCH --account=def-ibenayed
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100_1g.10gb:1

#SBATCH hetjob

#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --gpus=h100_1g.10gb:8
set -x

SLURM_SCRIPTS_DIR=$HOME/MolGenDocking/slurm/baselines/heterogenous
# Split nodes
REWARD_NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
HEAD_NODE=${REWARD_NODES[0]}
WORKER_NODES=${REWARD_NODES[@]:1}
echo "Head node: $HEAD_NODE"
echo "Worker nodes: $WORKER_NODES"

# Get IP of head node
HEAD_NODE_IP=$(srun --nodes 1 --nodelist=$HEAD_NODE --ntasks=1 hostname --ip-address | head -n1)

# Launch head server
srun --nodes 1 --nodelist=$HEAD_NODE --ntasks=1 $SLURM_SCRIPTS_DIR/run_server.sh $HEAD_NODE_IP head

# Launch workers
for node in $WORKER_NODES; do
    srun --nodes 1 --nodelist=$node --ntasks=1 $SLURM_SCRIPTS_DIR/run_server.sh $HEAD_NODE_IP worker
done

srun --het-group=0 --ntasks=1 $SLURM_SCRIPTS_DIR/run_training.sh $HEAD_NODE_IP $1 $2

wait

echo "Training + reward server job finished."
