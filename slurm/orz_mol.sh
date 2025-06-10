#!/bin/bash
#SBATCH --job-name=orz_mol
#SBATCH --account=rrg-josedolz
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=48
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking

cp $SCRATCH/MolGenData/mol_orz_pocket_desc.zip $SLURM_TMPDIR
cd $SLURM_TMPDIR
unzip mol_orz.zip
cd $WORKING_DIR

module load cuda

export PATH=$HOME/qvina:$PATH
source $HOME/R1ENV/bin/activate

ray start --head
nvidia-smi
ray status


export ORZ_DATA_PATH=$SLURM_TMPDIR/mol_orz_pocket_desc
#export DEBUG_MODE=1
python -m mol_gen_docking.orz_mol
