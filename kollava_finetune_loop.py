import os
import json
import gc
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import Dataset as HFDataset
import numpy as np
import argparse
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    AutoModelForCausalLM,
    TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training, LoraConfig,
    get_peft_model, PeftModel
)

def load_and_process_data(json_files, images_dir):
    all_data = []
    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"[경고] {json_file}이 존재하지 않습니다.")
            continue
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"'{json_file}'에서 {len(data)}개 항목 로드됨")
            for item in data:
                if all(k in item for k in ['image_path', 'prompt', 'response']):
                    image_path = os.path.join(images_dir, item['image_path'])
                    if os.path.exists(image_path):
                        text = f"질문: {item['prompt']}\n답변: {item['response']}"
                        all_data.append({'image_path': image_path, 'text': text})
                    else:
                        print(f"[경고] 이미지 없음: {image_path}")
    print(f"총 {len(all_data)}개 샘플 수집 완료")
    return all_data

def create_dataset(data):
    return HFDataset.from_dict({
        'image_path': [item['image_path'] for item in data],
        'text': [item['text'] for item in data]
    })

def preprocess_function(examples, tokenizer, image_processor):
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(examples["text"])):
        try:
            image = Image.open(examples['image_path'][i]).convert('RGB')
            image_tensor = image_processor(images=image, return_tensors="pt")
            inputs = tokenizer(
                examples['text'][i],
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True
            )
            ids = inputs['input_ids'][0].tolist()
            mask = inputs['attention_mask'][0].tolist()
        except Exception as e:
            print(f"[오류] 전처리 실패 - {e} ({examples['image_path'][i]})")
            ids = [tokenizer.pad_token_id] * 512
            mask = [0] * 512
        result['input_ids'].append(ids)
        result['attention_mask'].append(mask)
        result['labels'].append(ids)
    return result

def setup_model_and_processor(model_name, checkpoint_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    image_processor = CLIPImageProcessor.from_pretrained("/root/KoLLaVA-KoVicuna-7b")

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"체크포인트 로드: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        print("체크포인트에서 모델 로드 완료")

        if sum(p.requires_grad for p in model.parameters()) == 0:
            print("훈련 가능한 파라미터가 없어 활성화합니다.")
            for name, param in model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
    else:
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model, tokenizer, image_processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_8bit", action="store_true", help="Enable 8-bit quantization for model loading")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--json_paths', nargs='+', required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    print("모델 및 Processor 설정 중...")
    model, tokenizer, image_processor = setup_model_and_processor(args.model_name, args.checkpoint_dir)

    data = load_and_process_data(args.json_paths, args.images_dir)
    dataset = create_dataset(data)
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, image_processor), batched=True, batch_size=4)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        save_strategy="steps",
        evaluation_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model(args.checkpoint_dir)
    print(f"모델 저장 완료: {args.checkpoint_dir}")

if __name__ == "__main__":
    main()
