#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=def-ibenayed
#SBATCH --time=01:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --gpus=h100:1
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=sair_processed_meeko

cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz

cd $WORKING_DIR
cp data/properties.csv $SLURM_TMPDIR

export DATA_PATH=$SLURM_TMPDIR/$DATASET
source $HOME/OpenRLHF/bin/activate
export PATH=$PATH:$HOME/autodock_vina_1_1_2_linux_x86/bin

ray start --head --node-ip-address 0.0.0.0

coverage run -m pytest test/test_rewards/test_docking_API_autodock_gpu.py

# Launch server
python mol_gen_docking/fast_api_reward_server.py \
  --data-path $DATA_PATH --port 5001 --host 0.0.0.0 \
  --scorer-ncpus 4 --docking-oracle soft_docking --scorer-exhaustivness 4 &
sleep 10

coverage run -m pytest test/test_rewards/test_docking_server_autodock_gpu.py -x -s

kill -9 $(lsof -t -i :5001)
