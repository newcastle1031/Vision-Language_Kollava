python3 /root/kollava_finetune_loop.py \
  --model_name tabtoyou/KoLLaVA-KoVicuna-7b \
  --json_paths /root/data/data_3.json \
  --images_dir /root/ko_images \
  --checkpoint_dir /root/checkpoints/final \
  --output_dir /root/checkpoints/output \
  --use_8bit