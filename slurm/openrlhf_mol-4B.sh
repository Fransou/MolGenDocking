#!/bin/bash
#SBATCH --job-name=orz_mol
#SBATCH --account=def-ibenayed
#SBATCH --time=00:30:00
#SBATCH --gpus=h100:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=mol_orz

cp $SCRATCH/MolGenData/$DATASET.zip $SLURM_TMPDIR
cd $SLURM_TMPDIR
unzip -q $DATASET.zip
cd $WORKING_DIR

module load cuda

export PATH=$HOME/autodock_vina_1_1_2_linux_x86/bin/vina:$PATH
export DATA_PATH=$SLURM_TMPDIR/$DATASET
source $HOME/OpenRLHF/bin/activate

ray start --head --node-ip-address 0.0.0.0

python -m mol_gen_docking.fast_api_reward_server --data-path $SLURM_TMPDIR/$DATASET &
sleep 15

wandb offline
#export DEBUG_MODE=1
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"setup_commands": ["export WANDB_MODE=offline"]}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.9 \
   --pretrain $SCRATCH/Qwen/Qwen3-4B \
   --remote_rm_url http://localhost:5000/get_reward \
   --save_path $SCRATCH/checkpoint \
   --micro_train_batch_size 2 \
   --train_batch_size 1024 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 512 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.0 \
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
   --use_wandb 95190474fa39dc888a012cd12b18ab9b094697ad
