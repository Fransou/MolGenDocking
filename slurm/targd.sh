#!/bin/bash
#SBATCH --job-name=targd
#SBATCH --account=def-ibenayed
#SBATCH --time=00:00:00
#SBATCH --gpus=h100:4
#SBATCH --mem=248G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=0-0


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

source $HOME/OpenRLHF/bin/activate
module load cuda/12.6

python slurm/create_pdb_pocket.py \
 $SLURM_ARRAY_TASK_ID \
 $SLURM_TMPDIR/pocket.pdb \
 --data_path $SCRATCH/MolGenData/molgendata


cd external_repositories/targetdiff
python3 -m scripts.sample_for_pocket configs/sampling.yml --pdb_path $SLURM_TMPDIR/pocket.pdb --result_path $SLURM_TMPDIR/outputs_pdb

obabel $SLURM_TMPDIR/outputs_pdb/sdf/*.sdf -osmi -O $SLURM_TMPDIR/tmp.smi

python slurm/process_result.py \
 $SLURM_ARRAY_TASK_ID \
 $SLURM_TMPDIR/tmp.smi \
 $SCRATCH/MolGenOutput/TargetDiff \
 --data_path $SCRATCH/MolGenData/molgendata

