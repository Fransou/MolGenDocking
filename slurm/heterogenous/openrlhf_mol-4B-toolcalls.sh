#!/bin/bash
#SBATCH --job-name=orz_mol-toolcalls
#SBATCH --account=def-ibenayed
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --mem=248G
#SBATCH --cpus-per-task=32
#SBATCH --gpus=h100:2

#SBATCH hetjob

#SBATCH --nodes=2
#SBATCH --mem=750G
#SBATCH --cpus-per-task=192

set -x

# Split nodes
CPU_NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_1))
HEAD_NODE=${CPU_NODES[0]}
WORKER_NODES=${CPU_NODES[@]:1}
echo "Head node: $HEAD_NODE"
echo "Worker nodes: $WORKER_NODES"

# Get IP of head node
HEAD_NODE_IP=$(srun --het-group=1 --nodes 1 --nodelist=$HEAD_NODE --ntasks=1 hostname --ip-address | head -n1)

# Launch head server
srun --het-group=1 --nodes 1 --nodelist=$HEAD_NODE --ntasks=1 $HOME/MolGenDocking/slurm/heterogenous/run_server.sh $HEAD_NODE_IP head &

# Launch workers
for node in $WORKER_NODES; do
    srun --het-group=1 --nodes 1 --nodelist=$node --ntasks=1 $HOME/MolGenDocking/slurm/heterogenous/run_server.sh $HEAD_NODE_IP worker &
done

srun --het-group=0 --ntasks=1 $HOME/MolGenDocking/slurm/heterogenous/run_training.sh $HEAD_NODE_IP $1 $2

wait

echo "Training + reward server job finished."