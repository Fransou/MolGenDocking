#!/bin/bash

echo "Starting training model server"
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


wandb offline
export GPUS_PER_NODES=2
export N_SAMPLES_PER_PROMPT=$2
export BATCH_SIZE=$3

export PRETRAIN=$SCRATCH/Qwen/sft_Qwen-4B/model

#export DEBUG_MODE=1
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"setup_commands": ["wandb offline"]}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node $GPUS_PER_NODES \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node $GPUS_PER_NODES \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node $GPUS_PER_NODES \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node $GPUS_PER_NODES \
   --vllm_num_engines $GPUS_PER_NODES \
   --vllm_tensor_parallel_size 1 \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.7 \
   --pretrain $PRETRAIN \
   --remote_rm_url http://$1:5000/get_reward \
   --save_path $SCRATCH/DockGen-4B-toolcalls-nsamples-$N_SAMPLES_PER_PROMPT-batchsize-$BATCH_SIZE \
   --ckpt_path $SCRATCH/checkpoint/DockGen-4B-toolcalls-nsamples-$N_SAMPLES_PER_PROMPT-batchsize-$BATCH_SIZE \
   --max_ckpt_num 5 \
   --save_steps 3 \
   --micro_train_batch_size $BATCH_SIZE \
   --train_batch_size $(($BATCH_SIZE * $GPUS_PER_NODES)) \
   --micro_rollout_batch_size $BATCH_SIZE \
   --rollout_batch_size $(($BATCH_SIZE * $GPUS_PER_NODES)) \
   --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 2560 \
   --generate_max_len 2048 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.1 \
   --advantage_estimator reinforce \
   --prompt_data $SLURM_TMPDIR/$DATASET/train_prompts \
   --input_key prompt \
   --apply_chat_template \
   --packing_samples \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --enforce_eager \
   --use_tool_calls \
   --use_wandb 95190474fa39dc888a012cd12b18ab9b094697ad
