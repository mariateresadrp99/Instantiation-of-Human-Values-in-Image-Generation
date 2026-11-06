"""Generate N images per value prompt with an open-weight T2I model (Sec. 3).

    python generate_images.py --model flux2 --out data/images/flux2
    python generate_images.py --model qwen  --out data/images/qwen
    python generate_images.py --model sd35  --out data/images/sd35

Resumable; one deterministic seed per image. Use --start/--end to split the CSV across GPUs.
"""

import argparse

import torch

from utils import image_dir, read_prompts

MODELS = {
    "flux2": dict(cls="Flux2KleinPipeline", id="black-forest-labs/FLUX.2-klein-base-9B",
                  kwargs=dict(num_inference_steps=50, guidance_scale=4.0, height=1024, width=1024)),
    "qwen": dict(cls="QwenImagePipeline", id="Qwen/Qwen-Image",
                 kwargs=dict(num_inference_steps=50, true_cfg_scale=4.0, negative_prompt=" ", height=1328, width=1328)),
    "sd35": dict(cls="StableDiffusion3Pipeline", id="stabilityai/stable-diffusion-3.5-medium",
                 kwargs=dict(num_inference_steps=40, guidance_scale=4.5, height=1024, width=1024)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="prompts.csv")
    ap.add_argument("--n", type=int, default=50, help="images per prompt")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    import diffusers
    cfg = MODELS[args.model]
    pipe = getattr(diffusers, cfg["cls"]).from_pretrained(cfg["id"], torch_dtype=torch.bfloat16).to("cuda")

    rows = read_prompts(args.csv)
    for row_idx in range(args.start, args.end or len(rows)):
        row = rows[row_idx]
        for col_idx, prompt_type in enumerate(["positive_prompt", "negative_prompt"]):
            out_dir = image_dir(args.out, row["schwartz_domain"], row["value"], prompt_type)
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(args.n):
                seed = 42 + 2_000_000 * row_idx + 1_000_000 * col_idx + i
                path = out_dir / f"{seed:010d}.png"
                if path.exists():
                    continue
                gen = torch.Generator("cuda").manual_seed(seed)
                pipe(prompt=row[prompt_type], generator=gen, **cfg["kwargs"]).images[0].save(path)
            print(f"[{row_idx}] {row['value']}/{prompt_type}: {row[prompt_type]}")


if __name__ == "__main__":
    main()
