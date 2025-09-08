#!/bin/bash

echo "Starting reward model server on IP:"
echo $1
echo "Role: $2"

source $HOME/.bashrc
source $HOME/OpenRLHF/bin/activate
port=6379

if [ "$2" == "head" ]; then
    # Start Ray head node
    ray start --head --node-ip-address=$1
    sleep 30
    ray status
    echo "Ray head node started."
elif [ "$2" == "worker" ]; then
    # Start Ray worker node and connect to the head node
    ray start --address=$1:$port --block
    echo "Ray worker node started."
fi

# Run the reward server only on the head node
if [ "$2" == "head" ]; then
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

    echo "Running reward server..."
    python -m mol_gen_docking.fast_api_reward_server \
      --data-path $SLURM_TMPDIR/$DATASET \
      --host $1
fi
