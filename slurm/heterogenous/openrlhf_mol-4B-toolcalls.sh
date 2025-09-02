#!/bin/bash
#SBATCH --job-name=orz_mol-toolcalls
#SBATCH --account=def-ibenayed
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:2

#SBATCH hetjob

#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=128
set -x

# Get IP address of the second node (reward server)
REWARD_NODE_IP=$(srun --het-group=1 --ntasks=1 hostname --ip-address | head -n1)

srun --het-group=1 --ntasks=1 ls -l $HOME/MolGenDocking/slurm/heterogenous


echo "Reward server will run on node with IP: $REWARD_NODE_IP"
srun --het-group=1 --ntasks=1 $HOME/MolGenDocking/slurm/heterogenous/run_server.sh $REWARD_NODE_IP &

sleep 30  # Wait for the server to start

# Run training on the first node (SAMPLES PER PROMPT, BATCH SIZE)
srun --het-group=0 --ntasks=1 $HOME/MolGenDocking/slurm/heterogenous/run_training.sh $REWARD_NODE_IP $1 $2

echo "Training + reward server job finished."