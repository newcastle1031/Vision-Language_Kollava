import os
import logging
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, CLIPImageProcessor, ProcessorMixin, LlavaProcessor
from peft import PeftModel

logger = logging.getLogger(__name__)

# ✅ 커스텀 Processor
class CustomLlavaProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, padding=None, truncation=None, max_length=None, return_tensors=None, **kwargs):
        result = {}
        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors)
            result.update(image_features)
        if text is not None:
            if isinstance(text, str):
                text = [text]
            text_features = self.tokenizer(
                text=text,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                **kwargs
            )
            result.update(text_features)
        return result

# ✅ Processor 생성기

def create_custom_processor(model_name):
    try:
        processor = LlavaProcessor.from_pretrained(model_name)
        processor.image_processor.size = {"height": 336, "width": 336}
        processor.image_processor.do_resize = True
        processor.image_processor.do_center_crop = False
        return processor
    except:
        image_processor = CLIPImageProcessor(
            size={"height": 336, "width": 336},
            do_resize=True,
            do_center_crop=False,
            do_normalize=True,
            do_rescale=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            rescale_factor=1/255
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if "<image>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        return CustomLlavaProcessor(image_processor, tokenizer)


def create_modified_llava_model(model_name, tokenizer, use_8bit=False, use_bf16=False):
    from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    quant_config = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    model.resize_token_embeddings(len(tokenizer))
    
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    return model

# ✅ 커스텀 Dataset
class LlavaDataset(TorchDataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ✅ 데이터 로드 함수

def load_dataset(json_path, image_dir):
    import json
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        img_path = os.path.join(image_dir, item.get("image"))
        if not os.path.exists(img_path):
            continue

        question, answer = None, None
        for conv in item.get("conversations", []):
            if conv.get("from") == "human":
                question = conv.get("value")
            elif conv.get("from") in ["gpt", "assistant"]:
                answer = conv.get("value")

        if question and answer:
            data.append({"image_path": img_path, "question": question, "answer": answer})

    return LlavaDataset(data)

class SimpleLlavaDataCollator:
    def __init__(self, processor, tokenizer, max_length=512):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        input_ids_list = []
        attention_masks = []
        pixel_values = []
        labels_list = []

        for ex in examples:
            # 이미지 불러오기
            image = Image.open(ex["image_path"]).convert("RGB")
            pixel_values.append(self.processor.image_processor(image, return_tensors="pt")["pixel_values"][0])

            # 텍스트 전처리
            question = ex["question"]
            answer = ex["answer"]

            if "<image>" not in question:
                question = "<image>\n" + question

            # input 및 label 생성
            input_ids = self.tokenizer(
                question,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )["input_ids"][0]

            label_ids = self.tokenizer(
                answer,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )["input_ids"][0]

            # <image> 토큰 인덱스가 반드시 있어야 함
            image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
            if image_token_id not in input_ids:
                raise ValueError(f"<image> token ID ({image_token_id}) not found in input_ids: {input_ids}")

            input_ids_list.append(input_ids)
            attention_masks.append(input_ids.ne(self.tokenizer.pad_token_id).long())
            labels_list.append(label_ids)

        # 텐서 변환
        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_masks),
            "pixel_values": torch.stack(pixel_values),
            "labels": torch.stack(labels_list)
        }


# ✅ 커스텀 Trainer
from transformers import Trainer
class LlavaTrainer(Trainer):
    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True
        )
