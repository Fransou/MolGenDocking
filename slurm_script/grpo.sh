#!/bin/bash
#SBATCH --job-name=distill_mol
#SBATCH --account=def-ibenayed
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

export PATH=$PATH:$HOME/autodock_vina_1_1_2_linux_x86/bin
export PATH=$PATH:$HOME/ADFRsuite-1.0/bin
export WORKING_DIR=$HOME/MolGen/MolGenDocking
export SLURM_DIR=$SLURM_TMPDIR/tmp_dir

module load python/3.10 scipy-stack
module load openmm rdkit openbabel arrow
source $HOME/R1ENV/bin/activate

echo "Starting job on dataset"

mkdir $SLURM_DIR
cd $SLURM_DIR
cp -r $WORKING_DIR SLURM_DIR
cd MolGenDocking

wandb offline
python grpo.py
