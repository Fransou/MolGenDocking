#!/bin/bash
#SBATCH --job-name=orz_mol-toolcalls
#SBATCH --account=def-ibenayed
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --mem=248G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=h100:2

#SBATCH hetjob

#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=128

set -x
SLURM_SCRIPTS_DIR=$HOME/MolGenDocking/slurm/openrlhf

# Split nodes
CPU_NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
HEAD_NODE=${CPU_NODES[0]}
WORKER_NODES=${CPU_NODES[@]:1}
echo "Head node: $HEAD_NODE"
echo "Worker nodes: $WORKER_NODES"

# Get IP of head node
HEAD_NODE_IP=$(srun --nodes 1 --nodelist=$HEAD_NODE --ntasks=1 hostname --ip-address | head -n1)

# Launch head server
srun --nodes 1 --nodelist=$HEAD_NODE --ntasks=1 $HOME/MolGenDocking/slurm/heterogenous/run_server.sh $HEAD_NODE_IP head

# Launch workers
for node in $WORKER_NODES; do
    srun --nodes 1 --nodelist=$node --ntasks=1 $SLURM_SCRIPTS_DIR/run_server.sh $HEAD_NODE_IP worker
done

srun --het-group=0 --ntasks=1 $SLURM_SCRIPTS_DIR/heterogenous/run_training.sh $HEAD_NODE_IP $1 $2

wait

echo "Training + reward server job finished."
