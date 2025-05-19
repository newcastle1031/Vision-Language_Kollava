import os
import json
import argparse
import torch
from transformers import (
    AutoTokenizer, AutoConfig, 
    Trainer, TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset

# LLaVA 모델 직접 임포트 시도
try:
    from transformers import LlavaForConditionalGeneration, LlavaConfig
    print("성공적으로 transformers에서 LLaVA 모델을 임포트했습니다.")
    USE_TRANSFORMERS_LLAVA = True
except ImportError:
    print("transformers에서 LLaVA 모델을 찾을 수 없습니다. 다른 방법을 시도합니다.")
    USE_TRANSFORMERS_LLAVA = False
    try:
        # LLaVA 디렉토리에서 임포트 시도
        import sys
        sys.path.append('/root/LLaVA')
        from llava.model.llava_llama import LlavaLlamaForCausalLM
        from llava.model.llava_config import LlavaConfig
        print("성공적으로 로컬 LLaVA 모듈을 임포트했습니다.")
    except ImportError:
        print("로컬 LLaVA 모듈도 찾을 수 없습니다. tabtoyou/KoLLaVA-KoVicuna-7b 모델 구조에 맞는 클래스를 사용해야 합니다.")
        print("KoLLaVA 저장소를 확인하고 필요한 모듈을 설치하세요.")
        raise

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--deepspeed_config", type=str, required=True)
    
    # LoRA params
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training params
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--local_rank", type=int, default=-1, help="for deepspeed")

    return parser.parse_args()

def load_dataset(json_path, image_dir):
    with open(json_path, 'r') as f:
        raw = json.load(f)
    data = []
    for entry in raw:
        img_path = os.path.join(image_dir, entry["image"])
        for conv in entry["conversations"]:
            if conv["from"] == "human":
                question = conv["value"]
            else:
                data.append({
                    "image": img_path,
                    "input": f"<image>\n{question}",
                    "output": conv["value"]
                })
    return Dataset.from_list(data)

def tokenize_function(example, tokenizer):
    inputs = tokenizer(example["input"], truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(example["output"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

def main():
    args = parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    print("Loading model configuration...")
    config = AutoConfig.from_pretrained(args.model_name)
    print(f"모델 구성 유형: {type(config)}")

    if args.use_8bit:
        print("8-bit quantization enabled")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    else:
        quantization_config = None

    print("Loading base model...")
    if USE_TRANSFORMERS_LLAVA:
        print("transformers의 LlavaForConditionalGeneration 사용")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name,
            quantization_config=quantization_config if args.use_8bit else None,
            torch_dtype=torch.float16 if args.use_bf16 else torch.float32,
            low_cpu_mem_usage=True
        )
    else:
        print("LlavaLlamaForCausalLM 사용")
        model = LlavaLlamaForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config if args.use_8bit else None,
            torch_dtype=torch.float16 if args.use_bf16 else torch.float32,
            low_cpu_mem_usage=True
        )
    
    print("모델 로드 완료, 파라미터 수:", sum(p.numel() for p in model.parameters()))

    print("Preparing model for PEFT training...")
    print("Configuring LoRA...")
    #model = prepare_model_for_kbit_training(model)
    print("Configuring LoRA...")
    target_modules = [
    "q_proj", "v_proj", "k_proj", "o_proj",  
    ]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )
    model = get_peft_model(model, peft_config)

    print("Loading dataset...")
    dataset = load_dataset(args.data_path, args.image_dir)
    print(f"Dataset size: {len(dataset)} examples")
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer))

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,       
        save_strategy="epoch",              
        save_total_limit=3,                   
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,                       
        bf16=args.use_bf16 if hasattr(args, "use_bf16") else False,
        fp16=args.use_fp16 if hasattr(args, "use_fp16") else False,
        report_to="none",                   
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    '''
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        #gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=args.use_bf16,
        save_strategy="yes",
        logging_dir=f"{args.output_dir}/logs",
        deepspeed=args.deepspeed_config,
        report_to="none"
    )

    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
'''
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained(os.path.join(args.checkpoint_dir, "checkpoint"))
    print("Training completed successfully!")
  


if __name__ == "__main__":
    main()