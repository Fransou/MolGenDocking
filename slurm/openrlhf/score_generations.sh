#!/bin/bash
#SBATCH --job-name=scoring_compl
#SBATCH --account=def-ibenayed
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --mem=248G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=h100:4

export WORKING_DIR=$HOME/MolGenDocking

source $HOME/.bashrc
source $HOME/OpenRLHF/bin/activate

export DATASET=molgendata
cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz
cd $WORKING_DIR
cp data/properties.csv $SLURM_TMPDIR
export DATA_PATH=$SLURM_TMPDIR/$DATASET

ray start --head

export docking_oracle=autodock_gpu
export scorer_exhaustiveness=4

for input_file in "$1"/*; do
    if [ -f "$input_file" ]; then
        echo "Processing file: $input_file"
        python -m mol_gen_docking.score_completions \
            --input_file "$input_file" \
            --batch_size 1024
        echo "Finished scoring completions for $input_file."
    fi
done
