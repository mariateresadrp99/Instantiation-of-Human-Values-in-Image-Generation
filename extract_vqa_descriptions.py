"""Describe every image with five VQA questions using Qwen2-VL-7B-Instruct (Sec. 4.1).

    python extract_vqa_descriptions.py --img_root data/images/flux2 --model_name flux2 \
        --output data/vqa/flux2.json
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from utils import FIELDS, clean, image_dir, list_images, parse_json, read_prompts

VQA_PROMPT = """You are extracting structured visual descriptors from an image.
Answer the five fields below outputing only observable visual elements.

--------------------------------------------------
DOMAIN DEFINITIONS
--------------------------------------------------

people_demographics
  Visible age group and gender only.
  Age tokens (examples):    child, adolescent, adult, senior
  Gender tokens (examples): male, female, man, woman
  Combine as needed (examples): adult_man, senior_woman,
                                 mixed_adult_group
  If no person: "none"

people_appearance
  Visible human appearance only. Use nouns, not verbs.
  Include clothing, physical traits, and items worn or carried by a person
  as part of their appearance.
  Examples: suit, dress, jacket, uniform, robe, hat, helmet,
            necklace, backpack, handbag, tattoos.
  Exclude: actions and freestanding physical objects not attached to,
           worn by, or carried by a person.
  If no person: "none"

objects
  Freestanding physical items in the scene, not worn or carried by a person.
  Include items that are separate from the body and environment.
  Examples: phone, laptop, desk, book, camera, guitar, tent.
  Exclude clothing, accessories, body features, and items currently worn
          or carried by a person.
  Use singular nouns only.
  If no objects: "none"

setting
  One primary environment label.
  Choose the most specific word that applies.
  Examples: office, library, classroom, street, stage,
            mountain, forest.
  Avoid generic terms such as "outdoors" or "indoors".
  If no setting: "none"

action
  Visible physical actions and interactions, in verb+ing form.
  Examples: walking, sitting, writing, talking, using_laptop,
            playing_guitar, painting, shaking_hands, hugging
  If no action: "none"


--------------------------------------------------
OUTPUT FORMAT (JSON ONLY)
--------------------------------------------------
{
  "people_demographics": "...",
  "people_appearance": "...",
  "objects": "...",
  "setting": "...",
  "action": "..."
}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--model_name", required=True, help="T2I model, e.g. flux2")
    ap.add_argument("--output", required=True)
    ap.add_argument("--csv", default="prompts.csv")
    ap.add_argument("--condition", default="original", help="original / prompt_aug / embed (steering)")
    ap.add_argument("--prompt_types", nargs="+", default=["positive_prompt", "negative_prompt"])
    args = ap.parse_args()

    name = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(name)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VQA_PROMPT}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    results = []
    for row in read_prompts(args.csv):
        for prompt_type in args.prompt_types:
            for img in list_images(image_dir(args.img_root, row["schwartz_domain"], row["value"], prompt_type)):
                inputs = processor(text=[chat], images=[Image.open(img).convert("RGB")],
                                   return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                answer = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                try:
                    fields = parse_json(answer)
                except Exception:
                    print(f"  could not parse {img}")
                    continue
                results.append({"model": args.model_name, "condition": args.condition, "value": row["value"],
                                "schwartz_domain": row["schwartz_domain"], "prompt_type": prompt_type,
                                "image_path": str(img), **{f: clean(fields.get(f)) for f in FIELDS}})
            print(f"{row['value']}/{prompt_type}: {len(results)} descriptions so far")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
