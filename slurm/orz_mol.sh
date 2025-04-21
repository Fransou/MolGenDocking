#!/bin/bash
#SBATCH --job-name=orz_mol
#SBATCH --account=rrg-josedolz
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:3
#SBATCH --mem=150G
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

ray start --head --num-cpus 48

export DEBUG_MODE=1
python -m mol_gen_docking.orz_mol
