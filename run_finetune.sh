
for i in {1..2}
do
  echo "=== 학습 시작: data_$i.json ==="

  deepspeed --include localhost:0 --master_port=29505 finetune_kollava.py \
    --model_name "tabtoyou/KoLLaVA-KoVicuna-7b" \
    --data_path "/root/ko_data/data_${i}.json" \
    --image_dir "/root/ko_images" \
    --output_dir "/root/output/data_${i}" \
    --checkpoint_dir "/root/checkpoints/final" \
    --deepspeed_config "/root/ds_config.json" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --use_bf16 \
    --use_8bit
done

