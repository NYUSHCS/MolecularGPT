

deepspeed  ./finetune_moleculargpt.py \
    --model_name_or_path ./ckpts/llama \
    --train_files ./train_dataset/0-4-shot \
    --validation_files ./test_dataset/0-shot/bace/0.json \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir ./ckpts/lora \
    --evaluation_strategy  steps \
    --max_eval_samples 800 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 64 \
    --num_train_epochs 3 \
    --warmup_steps  400 \
    --load_in_bits 4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --target_modules q_proj,k_proj,v_proj,down_proj,up_proj \
    --logging_dir ./ckpts/lora/logs \
    --logging_strategy steps \
    --logging_steps 20 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 928 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 512 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --deepspeed ./ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 1800000000 \


    # --resume_from_checkpoint ./ckpts/lora/checkpoint- \
