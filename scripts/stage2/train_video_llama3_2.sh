#!/bin/bash

PREV_STAGE_CHECKPOINT="./checkpoints/cambrian_Llama3_2_3B_img"
PATH_TO_JSON=""
PATH_TO_FOLDER=""
VERSION="llama3"

CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=8 \
tdc/train.py \
--output_dir "./" \
--input_model_filename $PREV_STAGE_CHECKPOINT \
--output_model_filename "./checkpoints/mm_cambrian_llama3_2_video" \
--data_path $PATH_TO_JSON \
--image_folder $PATH_TO_FOLDER \
--model_max_length 8192 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir ./train/ \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--save_steps 1000 \
--eval_steps 5000 \
--logging_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--tf32 False \
--version $VERSION \
--mm_vision_select_layer "-2" \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--dataloader_num_workers 2 \
--lazy_preprocess True \
--tune_mm_mlp_adapter False \
--freeze_mm_mlp_adapter False \
--freeze_backbone False \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
--gradient_checkpointing True \
--mm_projector_type sva \
--image_token_len 144 \
--query_num_list "[144]" \
--resume-from-checkpoint True \
--lowres_token 8 \
--video_fps 1 \
--highres True \
--dino_threshold 0.9 \
--pretrained_qformer ./checkpoints/bert-base-uncased \
--audio_input False \
--max_num_segments 24 \
--text_input True \
--query_type Avg_pool \
--context_token_num 16 \
--add_static True