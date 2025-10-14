#!/bin/bash
#SBATCH --job-name=meeko_preprocess
#SBATCH --account=def-ibenayed
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=192
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=sair_processed

cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz

cd $WORKING_DIR

export PATH=$HOME/autodock_vina_1_1_2_linux_x86/bin/vina:$PATH
export DATA_PATH=$SLURM_TMPDIR/$DATASET
source $HOME/OpenRLHF/bin/activate

ray start --head --node-ip-address 0.0.0.0
python mol_gen_docking/data/meeko_process.py --data_path $DATA_PATH

cd $SLURM_TMPDIR
tar -czf processed_$DATASET.tar $DATASET

cp processed_$DATASET.tar $SCRATCH/MolGenData/
