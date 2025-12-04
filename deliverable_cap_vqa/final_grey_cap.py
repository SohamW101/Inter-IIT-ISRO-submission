import os
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def load_model():
    MODEL_DIR = "deliverable_cap_vqa/qwen3vl_final_greyscale_ft_cap"    
    BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  

    print("[LOAD] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("[LOAD] Loading fine-tuned Qwen3-VL model...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()

    return processor, model


def generate_caption(image_path, query, processor, model):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{query}"}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_len:]

    caption = processor.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    return caption


if __name__ == "__main__":
    IMAGE_PATH = "sar_bmd_2.png"   # <-- change this

    print(f"[RUN] Generating caption for: {IMAGE_PATH}")
    caption = generate_caption(IMAGE_PATH)

    print("\n=============================")
    print("📌 PREDICTED CAPTION:")
    print("=============================")
    print(caption)
    print("=============================\n")
