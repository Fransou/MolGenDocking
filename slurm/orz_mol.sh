#!/bin/bash
#SBATCH --job-name=orz_mol
#SBATCH --account=def-ibenayed
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=mol_orz

cp $SCRATCH/MolGenData/$DATASET.zip $SLURM_TMPDIR
cd $SLURM_TMPDIR
unzip -q $DATASET.zip
cd $WORKING_DIR

module load cuda

export PATH=$HOME/qvina:$PATH
source $HOME/R1ENV_3.11/bin/activate

ray start --head
nvidia-smi
ray status


export ORZ_DATA_PATH=$SLURM_TMPDIR/$DATASET
#export DEBUG_MODE=1
python -m mol_gen_docking.orz_mol
