#!/bin/bash
#SBATCH --job-name=orz_mol
#SBATCH --account=def-ibenayed
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=192
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=0-100

export IDX=$SLURM_ARRAY_TASK_ID
export DATA_PATH=$SCRATCH/MolGenData/SAIR

# Get the tar.gz file corresponding IDX-th in the DATA_PATH/structures_compressed
export DATA_FILE = $(ls $DATA_PATH/structures_compressed | sed -n "${IDX}p")


source $HOME/.bashrc
source $HOME/OpenRLHF/bin/activate
export WORKING_DIR=$HOME/MolGenDocking

cd $SLURM_TMPDIR
cp $DATA_PATH/structures_compressed/$DATA_FILE $SLURM_TMPDIR
# tar the file into a folder named sair_$IDX
mkdir sair_$IDX
tar -xvzf $DATA_FILE -C sair_$IDX

ray start --head --node-ip-address 0.0.0.0

python mol_gen_docking/data/SAIR_identify_pockets.py \
  --sair-dir sair_$IDX \
  --iou-threshold 0.4 \
  --topk 3

# Copy results back to SCRATCH
cp -r sair_$IDX $DATA_PATH/structures_with_pockets/sair_$IDX
