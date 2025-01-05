#!/bin/bash

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version v1 \
    --data_path ./playground/data/train_dataset.json \
    --validation_data_path ./playground/data/ai4mars/vqa/eval_dataset.json ./playground/data/ai4mars/instruction/eval_dataset.json ./playground/data/scoti/vqa/eval_dataset.json ./playground/data/llava_instruct/instruction/eval_dataset.json \
./playground/data/space_science_QA/qa/eval_dataset.json \
./playground/data/ai4mars/vqa/terrain_comparison/eval_dataset.json \
    --image_folder ./playground \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-lora-v1-0p2 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --eval_strategy "steps" \
    --eval_steps 700 \
    --save_strategy "epoch" \
    --learning_rate 3e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
