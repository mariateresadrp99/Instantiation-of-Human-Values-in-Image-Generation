#!/usr/bin/env python3
"""
extract_vqa_descriptions.py
-----------------------------
Extracts structured visual descriptors from generated images using
Qwen2-VL-7B-Instruct in a VQA setup.

Outputs one JSON record per image with fields:
    value, schwartz_domain, prompt_type, prompt_text, image_path,
    people_demographics, people_age, people_appearance, objects, setting, action

Usage:
    python vqa/extract_vqa_descriptions.py \\
        --csv prompts.csv \\
        --img_root data/images/gemini \\
        --output data/vqa/results_gemini_positive.json \\
        --prompt_type positive_prompt

    # Process both polarities:
    for MODEL in gemini flux2 qwen sd35; do
      for POLARITY in positive_prompt negative_prompt; do
        python vqa/extract_vqa_descriptions.py \\
          --csv prompts.csv \\
          --img_root data/images/$MODEL \\
          --output data/vqa/results_${MODEL}_${POLARITY}.json \\
          --prompt_type $POLARITY
      done
    done
"""

import argparse
import json
import os
import re
import glob
import traceback

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

EXPECTED_KEYS = {"people_demographics", "people_age", "people_appearance", "objects", "setting", "action"}

SYSTEM_PROMPT = """You are extracting structured visual descriptors from an image.

Output ONLY observable elements.
DO NOT infer personality, status, symbolism, or meaning.

--------------------------------------------------
FIELD DEFINITIONS
--------------------------------------------------

people_demographics:
- Report gender and count for each group present.
- Format: <count>_<gender> (e.g., 1_male, 2_female)
- Allowed gender tokens: male, female
- Separate multiple groups with commas: 1_male, 2_female
- If no person → "none".

people_age:
- Report age group and count for each group present.
- Format: <count>_<age> (e.g., 1_adult, 2_elder)
- Allowed age tokens: toddler, young_adult, adult, elder, unknown
- Separate multiple groups with commas: 1_adult, 1_elder
- If no person → "none".

people_appearance:
- ONLY visible human appearance. Use NO verbs.
- Include: clothing nouns (suit, dress, jacket, uniform, robe),
           worn accessories (necklace, backpack), physical traits.
- NEVER include actions or freestanding objects.
- If no person → "none".

objects:
- Physical items NOT worn on the body.
- Examples: phone, laptop, desk, book, camera, guitar, tent
- Use singular nouns only. Exclude clothing and accessories.
- If no objects → "none".

setting:
- ONE primary environment label only. Most specific single word.
- Examples: office, library, classroom, street, stage, mountain, forest
- Avoid generic words like "outdoors" or "indoors".
- If no setting → "none".

action:
- Visible physical actions only. Use verb+ing form.
- Examples: walking, sitting, writing, talking, using_laptop, painting, climbing
- If no action → "none".

--------------------------------------------------
GLOBAL RULES
--------------------------------------------------
- Output comma-separated tokens, lowercase, underscores instead of spaces.
- No duplicate words.

--------------------------------------------------
OUTPUT FORMAT (JSON ONLY)
--------------------------------------------------

{
  "people_demographics": "...",
  "people_appearance": "...",
  "people_age": "...",
  "objects": "...",
  "setting": "...",
  "action": "..."
}"""

USER_PROMPT = "Analyze this image and return the JSON tags."


# ── Model helpers ──────────────────────────────────────────────────────────────

def load_model(model_name: str):
    print(f"[LOAD] {model_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def run_inference(model, processor, image_path: str) -> dict:
    image    = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": USER_PROMPT},
        ]},
    ]

    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        parsed = json.loads(match.group())
        for k in EXPECTED_KEYS:
            parsed.setdefault(k, "none")
        return parsed
    return {"raw_output": raw, "parse_error": "No JSON found", **{k: "none" for k in EXPECTED_KEYS}}


def get_images(folder: str) -> list:
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        images.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(images)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract structured VQA descriptors from generated images."
    )
    ap.add_argument("--csv",         required=True, help="Path to prompts CSV")
    ap.add_argument("--img_root",    required=True, help="Root directory of generated images")
    ap.add_argument("--output",      required=True, help="Path to output JSON file")
    ap.add_argument("--prompt_type", default="positive_prompt",
                    choices=["positive_prompt", "negative_prompt"])
    ap.add_argument("--model",       default="Qwen/Qwen2-VL-7B-Instruct")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, sep=None, engine="python", index_col=0)
    df.columns = df.columns.str.strip()
    df.index   = df.index.str.strip()

    model, processor = load_model(args.model)
    results = []

    for idx, row in df.iterrows():
        value  = row.name
        domain = row["schwartz_domain"]
        folder = os.path.join(args.img_root, domain, value, args.prompt_type)

        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue

        images = get_images(folder)
        if not images:
            print(f"[WARN] No images in: {folder}")
            continue

        print(f"\n[{idx}] {domain}/{value}/{args.prompt_type} — {len(images)} images")

        for img_path in images:
            img_name = os.path.basename(img_path)
            print(f"  → {img_name}", end=" ", flush=True)

            try:
                tags   = run_inference(model, processor, img_path)
                status = "ok"
                print("✓")
            except Exception as e:
                tags   = {k: "none" for k in EXPECTED_KEYS}
                tags["error"] = str(e)
                status = "error"
                print(f"✗ {e}")
                traceback.print_exc()

            results.append({
                "value":          value,
                "schwartz_domain": domain,
                "prompt_type":    args.prompt_type,
                "prompt_text":    row.get(args.prompt_type, ""),
                "image_path":     img_path,
                "image_name":     img_name,
                "status":         status,
                **tags,
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] {len(results)} records → {args.output}")


if __name__ == "__main__":
    main()
