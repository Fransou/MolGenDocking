#!/bin/bash
#SBATCH --job-name=orz_mol
#SBATCH --account=rrg-josedolz
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking

cd $WORKING_DIR

module load cuda

export PATH=$HOME/qvina:$PATH
source $HOME/R1ENV/bin/activate

ray start --head
nvidia-smi
ray status

#export DEBUG_MODE=1
python -m mol_gen_docking.orz_mol
