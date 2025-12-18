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

i=$SLURM_ARRAY_TASK_ID
data_path=$SCRATCH/MolGenData/sair_data
files=($data_path/structures_compressed/*)

data_file=${files[$i]}


source $HOME/.bashrc
source $HOME/OpenRLHF/bin/activate
export WORKING_DIR=$HOME/MolGenDocking

cd $SLURM_TMPDIR
cp $data_file $SLURM_TMPDIR
mkdir sair_$i
tar -xzf $(basename $data_file) -C sair_$i
cp $data_path/sair.parquet $SLURM_TMPDIR/sair_$i

ray start --head --node-ip-address 0.0.0.0

cd $WORKING_DIR
python mol_gen_docking/data/SAIR_identify_pockets.py \
  --sair-dir $SLURM_TMPDIR/sair_$i \
  --output-dir $SLURM_TMPDIR/sair_pockets_$i \
  --iou-threshold 0.4 \
  --topk 3

# Copy results back to SCRATCH
mkdir $data_path/structures_with_pockets/sair_$i
cp -r $SLURM_TMPDIR/sair_pockets_$i $data_path/structures_with_pockets/sair_$i
