#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=def-ibenayed
#SBATCH --time=12:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=192
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=molgendata

cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz

cd $WORKING_DIR
cp data/properties.csv $SLURM_TMPDIR

export DATA_PATH=$SLURM_TMPDIR/$DATASET
source $HOME/OpenRLHF/bin/activate
module load autodock-vina/1.2.6


ray start --head --node-ip-address 0.0.0.0

pytest test/test_rewards/test_docking_API.py --accelerator=cpu
