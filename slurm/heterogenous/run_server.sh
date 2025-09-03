#!/bin/bash

echo "Starting reward model server on IP:"
echo $1

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=mol_orz

cp $SCRATCH/MolGenData/$DATASET.zip $SLURM_TMPDIR
cd $SLURM_TMPDIR
unzip -q $DATASET.zip

cd $WORKING_DIR
cp data/properties.csv $SLURM_TMPDIR

module load cuda

export PATH=$HOME/autodock_vina_1_1_2_linux_x86/bin/vina:$PATH
export DATA_PATH=$SLURM_TMPDIR/$DATASET
source $HOME/OpenRLHF/bin/activate

ray start --head --node-ip-address 0.0.0.0

python -m mol_gen_docking.fast_api_reward_server \
  --data-path $SLURM_TMPDIR/$DATASET \
  --host $1