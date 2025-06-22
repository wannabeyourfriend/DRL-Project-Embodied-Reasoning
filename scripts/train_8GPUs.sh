export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
MY_CUSTOM_PLUGINS="/cluster/home1/wzx/EgoReasoner/train/reward/imitation/reward.py"
CONSOLIDATED_DATASET_PATH="/cluster/home1/wzx/EgoReasoner/data/imitation/embodied_agent_train_dataset_1_100_s_consolidated_modified.jsonl" 

NPROC_PER_NODE=8 # Number of GPUs for training
NUM_INFER_WORKERS=1 # Number of GPUs for inference workers

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ----- Note the change here: Using torchrun -----
torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=1 \
/cluster/home2/yueyang/miniconda3/envs/swift_env/lib/python3.10/site-packages/swift/cli/rlhf.py \
    --rlhf_type grpo \
    --model "/cluster/home1/wzx/EgoReasoner/sft_weight" \
    --external_plugins "${MY_CUSTOM_PLUGINS}" \
    --reward_funcs 'comprehensive_decision_reward' \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.8 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset "${CONSOLIDATED_DATASET_PATH}" \
    --vllm_max_model_len 8192 \
    --max_completion_length 1024 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --output_dir "checkpoint/imitation" \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 0.7 \
    --repetition_penalty 1.0 \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers ${NUM_INFER_WORKERS} \
    --async_generate false \
    --beta 0.001 \
    --max_grad_norm 0.5 \
    --model_type 'qwen2_vl'
    # --num_generations 14
    # --max_model_len 32768 \
    # --limit-mm-per-prompt '{"image": 30, "video": 0}'
    #     --report_to wandb \