export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

pip3 install moviepy
mkdir videos
pip3 install huggingface_hub==0.24.0 
pip3 install tf-keras==2.19.0
pip3 install trl==0.16.0
pip3 install transformers==4.46.1
pip3 install protobuf==5.29.5

###Actually, we don't use the original pytorch torchrun in our internal environment, 
###so I just follow the official example of pytorch.
###Please adapt the torchrun scripts into your own environment
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=0 --master_addr=192.168.0.101 --master_port=29500 \
    fastvideo/train_grpo_hunyuan.py \
    --seed 42 \
    --model_type "hunyuan_hf" \
    --pretrained_model_name_or_path data/HunyuanVideo \
    --vae_model_path data/HunyuanVideo \
    --cache_dir data/.cache \
    --data_json_path data/rl_embeddings/videos2caption.json \
    --validation_prompt_dir data/Mochi-Black-Myth/validation \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 202 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --validation_steps 100000000 \
    --validation_sampling_steps 8 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir data/outputs/grpo \
    --tracker_project_name grpo \
    --h 480 \
    --w 480 \
    --t 53 \
    --sampling_steps 16 \
    --eta 0.25 \
    --lr_warmup_steps 0 \
    --fps 8 \
    --sampler_seed 1237 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 24 \
    --shift 5 \
    --use_group \
    --use_videoalign \
    --timestep_fraction 0.6 \
    --use_same_noise \
    --bestofn 8 \
    --vq_coef 1.0 \
    --mq_coef 0.0 