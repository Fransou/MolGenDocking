#!/bin/bash
#SBATCH --job-name=train_orz_mol
#SBATCH --account=rrg-josedolz
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:3
#SBATCH --mem=200G
#SBATCH --cpus-per-task=48
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

export VLLM_USE_V1=0

python -m mol_gen_docking.orz_mol
