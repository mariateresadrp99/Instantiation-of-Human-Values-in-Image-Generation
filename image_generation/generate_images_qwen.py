#!/usr/bin/env python3
"""
generate_images_qwen.py
------------------------
Generates images for Schwartz value prompts using Qwen-Image.
Resumable: skips already-generated images and tops up to the target count.

Usage:
    python generation/generate_images_qwen.py \\
        --csv prompts.csv \\
        --out data/images/qwen \\
        --images-per-prompt 50 \\
        --aspect-ratio 1:1 \\
        --which both
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from diffusers import DiffusionPipeline

DEFAULT_MODEL_ID = "Qwen/Qwen-Image"
QUALITY_SUFFIX   = ", Ultra HD, 4K"

# Qwen-Image optimized resolutions (width, height)
ASPECT_RATIOS = {
    "1:1":  (1328, 1328),
    "16:9": (1664,  928),
    "9:16": ( 928, 1664),
    "4:3":  (1472, 1140),
    "3:4":  (1140, 1472),
    "3:2":  (1584, 1056),
    "2:3":  (1056, 1584),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_slug(s: str, max_len: int = 120) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-. ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s[:max_len] if len(s) > max_len else s


def read_schwartz_csv(csv_path: Path) -> List[Dict[str, str]]:
    required = {"value", "schwartz_domain", "positive_prompt", "negative_prompt"}
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {missing}. Found: {reader.fieldnames}")
        for r in reader:
            if not (r.get("value") and r.get("schwartz_domain")):
                continue
            rows.append(r)
    return rows


def count_existing_pngs(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.png"))


def apply_cache_dir(cache_dir: str) -> None:
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"]               = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["TRANSFORMERS_CACHE"]    = str(cache_root / "transformers")
    os.environ["DIFFUSERS_CACHE"]       = str(cache_root / "diffusers")
    for key in ["HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "DIFFUSERS_CACHE"]:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


# ── Core generation ────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_images_topup(
    pipe: DiffusionPipeline,
    prompt: str,
    negative_prompt: str,
    out_dir: Path,
    n_images: int,
    steps: int,
    guidance: float,
    base_seed: int,
    device_for_generator: str,
    batch_size: int,
    height: int,
    width: int,
):
    """
    Generate images until out_dir contains n_images PNGs.
    Already-existing images are never regenerated — safe to resume.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_cursor = 0
    pos_prompt  = prompt + QUALITY_SUFFIX
    neg_prompt  = negative_prompt if negative_prompt else " "

    while True:
        existing = count_existing_pngs(out_dir)
        if existing >= n_images:
            print(f"  [SKIP/DONE] {out_dir.name} — {existing}/{n_images} images exist.")
            return

        need = n_images - existing
        bsz  = min(batch_size, need)

        seeds_to_run, gens = [], []
        while len(seeds_to_run) < bsz:
            s = base_seed + seed_cursor
            seed_cursor += 1
            filename = out_dir / f"{s:010d}.png"
            if filename.exists():
                continue
            seeds_to_run.append(s)
            gens.append(torch.Generator(device=device_for_generator).manual_seed(s))

        result = pipe(
            prompt=[pos_prompt] * len(seeds_to_run),
            negative_prompt=[neg_prompt] * len(seeds_to_run),
            height=height, width=width,
            true_cfg_scale=guidance,
            num_inference_steps=steps,
            generator=gens,
        )

        for j, img in enumerate(result.images):
            filename = out_dir / f"{seeds_to_run[j]:010d}.png"
            if not filename.exists():
                img.save(filename)
                print(f"    saved {filename.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Qwen-Image generator — resumable, slice-able, aspect-ratio aware."
    )
    ap.add_argument("--csv",               required=True,  help="Path to prompts CSV")
    ap.add_argument("--out",               required=True,  help="Output root directory")
    ap.add_argument("--model-id",          default=DEFAULT_MODEL_ID)
    ap.add_argument("--device",            default="cuda:0")
    ap.add_argument("--dtype",             choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--start-idx",         type=int, default=0)
    ap.add_argument("--end-idx",           type=int, default=None)
    ap.add_argument("--images-per-prompt", type=int, default=50)
    ap.add_argument("--steps",             type=int, default=50)
    ap.add_argument("--guidance",          type=float, default=4.0)
    ap.add_argument("--aspect-ratio",      choices=list(ASPECT_RATIOS.keys()), default="1:1")
    ap.add_argument("--base-seed",         type=int, default=42)
    ap.add_argument("--batch-size",        type=int, default=1)
    ap.add_argument("--which",             choices=["pos", "neg", "both"], default="both")
    ap.add_argument("--cpu-offload",       action="store_true")
    ap.add_argument("--cache-dir",         default=None)
    args = ap.parse_args()

    if args.cache_dir:
        apply_cache_dir(args.cache_dir)

    dtype_map       = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype     = dtype_map[args.dtype]
    target_w, target_h = ASPECT_RATIOS[args.aspect_ratio]

    print(f"[INFO] Aspect ratio {args.aspect_ratio} → {target_w}×{target_h}")

    rows    = read_schwartz_csv(Path(args.csv))
    end_idx = args.end_idx if args.end_idx is not None else len(rows)
    rows    = rows[args.start_idx:end_idx]
    print(f"[INFO] Processing rows {args.start_idx}–{end_idx} ({len(rows)} rows)")

    print(f"[LOAD] {args.model_id} ...")
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
        device_for_gen = "cuda" if "cuda" in args.device else "cpu"
    else:
        pipe = pipe.to(args.device)
        device_for_gen = "cuda" if "cuda" in args.device else "cpu"

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for local_idx, r in enumerate(rows):
        global_idx = args.start_idx + local_idx
        value      = safe_slug(r.get("value", ""))
        domain     = safe_slug(r.get("schwartz_domain", ""))
        pos_prompt = (r.get("positive_prompt") or "").strip()
        neg_prompt = (r.get("negative_prompt") or "").strip()

        base_dir      = out_root / domain / value
        row_seed_base = args.base_seed + 2_000_000 * global_idx

        print(f"[ROW {global_idx}] {domain}/{value}")

        if args.which in ("pos", "both") and pos_prompt:
            print(f"  POS: {pos_prompt[:100]}")
            generate_images_topup(
                pipe=pipe, prompt=pos_prompt, negative_prompt=neg_prompt,
                out_dir=base_dir / "positive_prompt",
                n_images=args.images_per_prompt, steps=args.steps,
                guidance=args.guidance, base_seed=row_seed_base,
                device_for_generator=device_for_gen,
                batch_size=max(1, args.batch_size),
                height=target_h, width=target_w,
            )

        if args.which in ("neg", "both") and neg_prompt:
            print(f"  NEG: {neg_prompt[:100]}")
            generate_images_topup(
                pipe=pipe, prompt=neg_prompt, negative_prompt=pos_prompt,
                out_dir=base_dir / "negative_prompt",
                n_images=args.images_per_prompt, steps=args.steps,
                guidance=args.guidance, base_seed=row_seed_base + 1_000_000,
                device_for_generator=device_for_gen,
                batch_size=max(1, args.batch_size),
                height=target_h, width=target_w,
            )

    print(f"\n[DONE] All images written to {out_root}")


if __name__ == "__main__":
    main()
