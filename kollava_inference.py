import torch
import os
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from peft import PeftModel
import argparse
import sys
sys.path.append("/root/KoLLaVA")
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

def main():
    parser = argparse.ArgumentParser(description="KoLLaVA 모델 인퍼런스 실행")
    parser.add_argument('--image_path', type=str, required=True, help='추론할 이미지 경로')
    parser.add_argument('--question', type=str, required=True, help='이미지에 대한 질문')
    parser.add_argument('--model_name', type=str, default='tabtoyou/KoLLaVA-KoVicuna-7b', help='KoLLaVA 베이스 모델 이름')
    parser.add_argument('--checkpoint_path', type=str, default='/root/checkpoints/final', help='LoRA 체크포인트 경로')
    args = parser.parse_args()

    # Tokenizer, ImageProcessor 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    image_processor = CLIPImageProcessor.from_pretrained("/root/KoLLaVA-KoVicuna-7b")

    # 모델 로드
    print("KoLLaVA 모델 로드 중...")
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("LoRA 체크포인트 적용 중...")
    model = PeftModel.from_pretrained(model, args.checkpoint_path)
    model.eval()

    # 이미지 및 프롬프트 준비
    image = Image.open(args.image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to("cuda")

    prompt = f"<image>\n{args.question}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # 모델 입력 포맷 구성
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print("KoLLaVA 추론 실행 중...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n🧠 KoLLaVA 응답:")
    print(response)

if __name__ == "__main__":
    main()
