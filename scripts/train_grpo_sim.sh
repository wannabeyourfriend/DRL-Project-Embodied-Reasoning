#!/bin/bash
# Configuration Description
# This script is designed for 4-card A800 80GB memory configuration

# Main function
main() {
    
    # Define system prompt
    SYSTEM_PROMPT="You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. Please follow the format: <think> [reasoning process] </think> <answer> [final answer] </answer>"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NPROC_PER_NODE=4 \
    swift rlhf \
    --rlhf_type grpo \
    --model "Qwen/Qwen2-VL-7B-Instruct" \
    --train_type full \
    --torch_dtype bfloat16 \
    --system "${SYSTEM_PROMPT}" \
    --dataset "path/to/dataset.jsonl" \
    --val_dataset "path/to/val_dataset.jsonl" \
    --dataloader_num_workers 4 \
    --num_train_epochs 2 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --target_modules all-linear \
    --reward_weights 0.8 0.2 \
    --reward_funcs plan_accuracy decision_format \
    --external_plugins /cluster/home1/wzx/EgoReasoner/train/reward/simulation/plan_accuracy_reward.py /cluster/home1/wzx/EgoReasoner/train/reward/format/decision_format_reward.py  \
    --beta 0.001 \
    --temperature 1.0 \
    --num_generations 4 \
    --num_iterations 1 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.6 \
    --tensor_parallel_size 4 \
    --deepspeed zero3 \
    --num_infer_workers 4 \
    --max_length 6144 \
    --max_completion_length 2048 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_only_model true \
    --save_total_limit 200 \
    --output_dir "results/train" \
    --logging_steps 1 \
    --report_to wandb \
    --log_completions true \
    --log_level debug \
    --log_level_replica warning \
    # 2>&1 | tee "results/train.log"
}
main