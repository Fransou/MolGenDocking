#!/bin/bash
#SBATCH --job-name=orz_mol
#SBATCH --account=def-ibenayed
#SBATCH --time=00:30:00
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=sair_processed_meeko

cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz
cd $WORKING_DIR

export PATH=$HOME/autodock_vina_1_1_2_linux_x86/bin/vina:$PATH
export DATA_PATH=$SLURM_TMPDIR/$DATASET
source $HOME/OpenRLHF/bin/activate

ray start --head --node-ip-address 0.0.0.0

export docking_oracle=autodock_gpu
export scorer_exhaustiveness=4
export docking_oracle=autodock_gpu
uvicorn --host 0.0.0.0 --port 5001 mol_gen_docking.server:app --log-level critical &

sleep 3

wandb offline
#export DEBUG_MODE=1
HF_HUB_OFFLINE=1 python -m mol_gen_docking.baselines.reinvent.rl_opt \
  --model_name reinvent_10M_prior \
  --dataset $DATA_PATH/eval_data/eval_prompts \
  --datasets-path $DATA_PATH \
  --batch_size 64 \
  --sigma 1 \
  --num_train_epochs 500 \
  --generation_config '{"num_beams": 2}' \
  --train_on_beams 0 \
  --id_obj 2
