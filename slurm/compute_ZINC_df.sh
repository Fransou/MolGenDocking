#!/bin/bash
#SBATCH --job-name=compute_ZINC_df
#SBATCH --account=rrg-josedolz
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

export WORKING_DIR=$HOME/MolGenDocking

r1
cd $WORKING_DIR
ray start --head --num-cpus 64

echo "Starting job on from $1 to $2"
python mol_gen_docking/compute_properties_ZINC.py \
  --batch-size 512 \
  --i_start $1 \
  --i_end $2
