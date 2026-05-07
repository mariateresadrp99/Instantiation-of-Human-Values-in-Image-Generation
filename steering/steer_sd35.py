#!/usr/bin/env python3
"""
steer_sd35.py
-------------
Text-embedding steering for Stable Diffusion 3.5 (Kim et al., CVPR 2025).

Injects an archetype direction into the post-EOS padding positions of the
T5-XXL prompt embedding, leaving content tokens unchanged.

Usage:
    python steering/steer_sd35.py \\
        --prompt "a successful person" \\
        --archetypes \\
            "a group of people of different ages celebrating at home" \\
            "a person in workwear at a construction site" \\
            "friends at a dinner table" \\
        --scale 1.0 \\
        --tau 0.4 \\
        --out results/steered/sd35/successful \\
        --seeds 42 43 44 45 46 47 48 49
"""

import argparse
import inspect
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"


# ── Embedding helpers ──────────────────────────────────────────────────────────

def get_text_embeddings(pipe, prompt: str, device, dtype):
    """Encode a prompt through all three SD3 text encoders."""
    result = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, prompt_3=prompt,
        device=device, num_images_per_prompt=1,
    )
    return result[0], result[2]  # (sequence_embeds, pooled_embeds)


def build_sd3_steered_embeds(pipe, base_prompt: str, arch_prompt: str,
                              device, dtype, scale: float):
    """
    Build steered sequence embedding using EOS-masked weak guidance.
        c_steered = c_base + M ⊙ scale * (c_arch − c_∅)
    M[i] = 1 for i ≥ first EOS/PAD position in T5 tokenizer.
    """
    c_base,  p_base  = get_text_embeddings(pipe, base_prompt,  device, dtype)
    c_arch,  _       = get_text_embeddings(pipe, arch_prompt,  device, dtype)
    c_empty, _       = get_text_embeddings(pipe, "",           device, dtype)

    direction = c_arch - c_empty

    # Find EOS in T5-XXL tokenizer (tokenizer_3)
    tokens    = pipe.tokenizer_3(
        base_prompt, padding="max_length", max_length=256,
        truncation=True, return_tensors="pt",
    )
    input_ids = tokens["input_ids"][0]
    t5_eos    = 256

    for token_id in [pipe.tokenizer_3.eos_token_id, pipe.tokenizer_3.pad_token_id]:
        if token_id is not None:
            pos = (input_ids == token_id).nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                t5_eos = min(t5_eos, int(pos[0].item()))

    # Map onto concatenated (CLIP-L + CLIP-G + T5) sequence
    seq_len          = c_base.shape[1]
    global_pad_start = (seq_len - 256) + t5_eos

    mask = torch.zeros(1, seq_len, 1, device=device, dtype=dtype)
    if global_pad_start < seq_len:
        mask[0, global_pad_start:, 0] = 1.0

    c_steered = c_base + mask * scale * direction
    return c_base, p_base, c_steered


# ── Denoising loop ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_handover(
    pipe, base_prompt: str, arch_prompt: str,
    seed: int, out_dir: Path,
    device, dtype, steps: int,
    scale: float, tau: float, guidance_scale: float,
    debug: bool,
):
    """
    Custom denoising loop with phased embedding handover:
      steps 0 … tau_step−1 : c_steered
      steps tau_step … T−1 : c_base
    """
    c_base, p_base, c_steered = build_sd3_steered_embeds(
        pipe, base_prompt, arch_prompt, device, dtype, scale
    )
    c_null, p_null = get_text_embeddings(pipe, "", device, dtype)

    tau_step  = int(tau * steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents   = pipe.prepare_latents(1, 16, 1024, 1024, dtype, device, generator)

    pipe.scheduler.set_timesteps(steps, device=device)
    call_sig = inspect.signature(pipe.transformer.forward)

    for i, t in enumerate(pipe.scheduler.timesteps):
        curr_c = c_steered if i < tau_step else c_base
        mode   = "steered" if i < tau_step else "base"

        if debug and (i < 2 or i == tau_step or i == steps - 1):
            print(f"  [STEP {i:2d}] t={float(t):7.2f} | {mode}")

        model_input = (
            pipe.scheduler.scale_model_input(latents, t)
            if hasattr(pipe.scheduler, "scale_model_input") else latents
        )

        def run_transformer(ec, ep):
            kw = dict(
                hidden_states=model_input,
                timestep=t.unsqueeze(0) if t.dim() == 0 else t,
                encoder_hidden_states=ec,
            )
            if "pooled_projections" in call_sig.parameters:
                kw["pooled_projections"] = ep
            return pipe.transformer(**kw).sample

        noise_pred = run_transformer(c_null, p_null) + guidance_scale * (
            run_transformer(curr_c, p_base) - run_transformer(c_null, p_null)
        )
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    scaling = pipe.vae.config.scaling_factor
    shift   = getattr(pipe.vae.config, "shift_factor", 0.0)
    latents = (latents - shift) / scaling

    image    = pipe.vae.decode(latents, return_dict=False)[0]
    image    = pipe.image_processor.postprocess(image, output_type="pil")[0]
    out_path = Path(out_dir) / f"seed_{seed}.png"
    image.save(out_path)
    print(f"  → {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SD3.5 text-embedding steering.")
    ap.add_argument("--prompt",         default="a successful person")
    ap.add_argument("--archetypes",     nargs="+", required=True)
    ap.add_argument("--out",            required=True)
    ap.add_argument("--seeds",          nargs="+", type=int, default=[42])
    ap.add_argument("--scale",          type=float, default=1.0)
    ap.add_argument("--tau",            type=float, default=0.4)
    ap.add_argument("--guidance-scale", type=float, default=7.0)
    ap.add_argument("--model-id",       default=DEFAULT_MODEL_ID)
    ap.add_argument("--debug",          action="store_true")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {args.model_id} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16
    ).to("cuda")

    for i, seed in enumerate(args.seeds):
        arch = args.archetypes[i % len(args.archetypes)]
        print(f"\n[GEN] seed {seed} | archetype: '{arch}'")
        generate_handover(
            pipe=pipe, base_prompt=args.prompt, arch_prompt=arch,
            seed=seed, out_dir=args.out,
            device="cuda", dtype=torch.bfloat16,
            steps=50, scale=args.scale, tau=args.tau,
            guidance_scale=args.guidance_scale, debug=args.debug,
        )


if __name__ == "__main__":
    main()
