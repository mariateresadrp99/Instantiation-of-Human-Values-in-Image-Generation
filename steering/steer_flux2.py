#!/usr/bin/env python3
"""
steer_flux2.py
--------------
Phased text-embedding steering for FLUX.2 Klein (Kim et al., CVPR 2025).

Steering method:
    c_hat = c_base + M ⊙ λ · (c_arch − c_∅)
                   − M ⊙ α · orth(c_away − c_∅, c_arch − c_∅)  [optional]
    M[i] = 1 for i ≥ EOS position (padding positions only).

Step schedule:
    steps 0 … floor(τ·T)     : c_hat only (steered)
    steps floor(τ·T) … T−1   : alternate c_hat / c_base

Usage:
    python steering/steer_flux2.py \\
        --prompt "a successful person" \\
        --archetypes \\
            "a group of people of different ages celebrating at home" \\
            "a person in workwear at a construction site" \\
            "friends at a dinner table" \\
        --scale 1.5 \\
        --tau 0.4 \\
        --out results/steered/flux2/successful \\
        --seeds 42 43 44 45 46 47 48 49 \\
        --debug
"""

import json
import math
import random
import argparse
import inspect
import os
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"


# ── Utilities ──────────────────────────────────────────────────────────────────

def apply_cache_dir(cache_dir: str) -> None:
    if not cache_dir:
        return
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"]               = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")


def get_text_embeddings(pipe, prompt: str, device, dtype):
    """Encode a prompt through FLUX's text encoders (T5 + CLIP)."""
    sig    = inspect.signature(pipe.encode_prompt)
    kwargs = dict(prompt=prompt, device=device, num_images_per_prompt=1)
    if "prompt_2" in sig.parameters:
        kwargs["prompt_2"] = prompt
    result = pipe.encode_prompt(**kwargs)
    if isinstance(result, (tuple, list)):
        return result[0], result[1]
    return result, None


def find_eos_position(pipe, prompt: str, debug: bool = False):
    """
    Return (eos_position, seq_len) from the T5 tokenizer (tokenizer_2).
    Injection starts at eos_position; real token positions are never modified.
    """
    tokenizer = getattr(pipe, "tokenizer_2", None) or pipe.tokenizer
    max_len   = min(getattr(tokenizer, "model_max_length", 512), 512)
    tokens    = tokenizer(prompt, padding="max_length", max_length=max_len,
                          truncation=True, return_tensors="pt")
    input_ids = tokens["input_ids"][0]
    seq_len   = input_ids.shape[0]
    eos_pos   = seq_len

    for token_id in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if token_id is None:
            continue
        positions = (input_ids == token_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            eos_pos = int(positions[0].item())
            break

    if debug:
        n_inject = seq_len - eos_pos
        print(f"  [EOS] seq_len={seq_len}, eos_pos={eos_pos}, inject_positions={n_inject}"
              + (" ⚠ very few" if n_inject < 5 else ""))

    return eos_pos, seq_len


def make_image_ids(height: int, width: int, vae_scale: int, device, dtype) -> torch.Tensor:
    """Build FLUX img_ids tensor with correct 2D RoPE positions."""
    latent_h = height // vae_scale // 2
    latent_w = width  // vae_scale // 2
    img_ids  = torch.zeros(latent_h, latent_w, 3, device=device, dtype=dtype)
    img_ids[..., 1] = torch.arange(latent_h, device=device).float()[:, None]
    img_ids[..., 2] = torch.arange(latent_w, device=device).float()[None, :]
    return img_ids.reshape(-1, 3)


def orthogonalise(v: torch.Tensor, against: torch.Tensor) -> torch.Tensor:
    """Remove the component of v that lies along `against`."""
    a_flat = against.reshape(-1).float()
    v_flat = v.reshape(-1).float()
    proj   = (v_flat @ a_flat) / (a_flat @ a_flat + 1e-8)
    return (v - proj * against).to(v.dtype)


# ── Embedding builder ──────────────────────────────────────────────────────────

def build_steered_embeds(
    pipe, base_prompt: str, archetype_prompt: str,
    device, dtype, scale: float = 1.0,
    away_prompt: str = None, away_scale: float = 0.5,
    debug: bool = False,
):
    """
    Build base and steered embeddings.
        c_hat = c_base + M ⊙ scale * (c_arch − c_∅)
                       − M ⊙ away_scale * orth(c_away − c_∅, c_arch − c_∅)
    """
    base_emb,  base_pooled = get_text_embeddings(pipe, base_prompt,     device, dtype)
    arch_emb,  _           = get_text_embeddings(pipe, archetype_prompt, device, dtype)
    empty_emb, _           = get_text_embeddings(pipe, "",               device, dtype)

    direction = arch_emb - empty_emb
    eos_pos, seq_len = find_eos_position(pipe, base_prompt, debug=debug)
    mask = torch.zeros(1, seq_len, 1, device=device, dtype=dtype)
    if eos_pos < seq_len:
        mask[0, eos_pos:, 0] = 1.0

    steered = base_emb + mask * scale * direction

    if away_prompt:
        away_emb, _ = get_text_embeddings(pipe, away_prompt, device, dtype)
        away_dir    = orthogonalise(away_emb - empty_emb, direction)
        steered     = steered - mask * away_scale * away_dir

    if debug:
        n_inj  = int(mask.sum().item())
        delta  = (steered - base_emb).norm().item()
        rel    = delta / (base_emb.norm().item() + 1e-8) * 100
        print(f"  [EMBED] arch='{archetype_prompt}' | injected={n_inj} | "
              f"Δnorm={delta:.4f} ({rel:.1f}% of base)")
        if n_inj == 0:
            print("  [EMBED] ⛔ 0 positions injected — steering is a NO-OP!")

    return base_emb, steered, base_pooled


# ── Phased denoising loop ──────────────────────────────────────────────────────

@torch.inference_mode()
def generate_phased(
    pipe, base_prompt: str, archetype_prompt: str,
    seed: int, out_dir: Path, device, dtype,
    steps: int, guidance: float, height: int, width: int,
    scale: float, tau: float,
    away_prompt: str, away_scale: float, debug: bool,
    prebuilt_embeds=None,
):
    if prebuilt_embeds is not None:
        c_hat, pooled = prebuilt_embeds
        c_base, pooled = get_text_embeddings(pipe, base_prompt, device, dtype)
    else:
        c_base, c_hat, pooled = build_steered_embeds(
            pipe, base_prompt, archetype_prompt, device, dtype,
            scale, away_prompt, away_scale, debug,
        )

    tau_step   = int(tau * steps)
    gen_device = "cuda" if "cuda" in str(device) else "cpu"
    generator  = torch.Generator(device=gen_device).manual_seed(seed)
    vae_scale  = getattr(pipe, "vae_scale_factor", 8)

    # Prepare latents
    try:
        n_ch    = pipe.transformer.config.in_channels // 4
        latents = pipe.prepare_latents(1, n_ch, height, width, dtype, device, generator)
    except Exception:
        in_ch   = getattr(pipe.transformer.config, "in_channels", 64)
        lh, lw  = height // vae_scale // 2, width // vae_scale // 2
        latents = torch.randn(1, in_ch, lh, lw, dtype=dtype, device=device, generator=generator)

    # Scheduler
    image_seq_len = (height // vae_scale // 2) * (width // vae_scale // 2)
    if hasattr(pipe, "_calculate_shift"):
        mu = pipe._calculate_shift(image_seq_len)
    else:
        base_seq   = getattr(pipe.scheduler.config, "base_image_seq_len", 256)
        max_seq    = getattr(pipe.scheduler.config, "max_image_seq_len",  4096)
        base_shift = getattr(pipe.scheduler.config, "base_shift",  0.5)
        max_shift  = getattr(pipe.scheduler.config, "max_shift",   1.15)
        mu = (math.log(max_shift / base_shift) *
              (image_seq_len - base_seq) / (max_seq - base_seq) +
              math.log(base_shift))

    set_ts_sig = inspect.signature(pipe.scheduler.set_timesteps)
    if "mu" in set_ts_sig.parameters:
        pipe.scheduler.set_timesteps(steps, device=device, mu=mu)
    else:
        pipe.scheduler.set_timesteps(steps, device=device)

    # Scale latents
    if hasattr(pipe.scheduler, "init_noise_sigma"):
        latents = latents * pipe.scheduler.init_noise_sigma
    else:
        latents = latents * pipe.scheduler.sigmas[0]

    img_ids_base = make_image_ids(height, width, vae_scale, device, dtype)
    call_sig     = inspect.signature(pipe.transformer.forward)

    # Denoising loop
    for step_idx, t in enumerate(pipe.scheduler.timesteps):
        curr = c_hat if (step_idx < tau_step or step_idx % 2 == 0) else c_base

        latent_input = (
            pipe.scheduler.scale_model_input(latents, t)
            if hasattr(pipe.scheduler, "scale_model_input") else latents
        )

        kwargs = dict(
            hidden_states=latent_input,
            timestep=t.unsqueeze(0) if t.dim() == 0 else t,
            encoder_hidden_states=curr,
            guidance=torch.tensor([guidance], device=device, dtype=dtype),
        )
        if "pooled_projections" in call_sig.parameters and pooled is not None:
            kwargs["pooled_projections"] = pooled
        if "txt_ids" in call_sig.parameters:
            kwargs["txt_ids"] = torch.zeros(curr.shape[1], 3, device=device, dtype=dtype)
        if "img_ids" in call_sig.parameters:
            kwargs["img_ids"] = img_ids_base

        noise_pred = pipe.transformer(**kwargs).sample
        latents    = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    if hasattr(pipe, "_unpack_latents"):
        latents = pipe._unpack_latents(latents, height, width, vae_scale)

    scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    shift   = getattr(pipe.vae.config, "shift_factor",   None)
    latents = (latents / scaling) + shift if shift is not None else latents / scaling

    image    = pipe.vae.decode(latents, return_dict=False)[0]
    image    = pipe.image_processor.postprocess(image, output_type="pil")[0]
    img_path = out_dir / f"seed_{seed}.png"
    image.save(img_path)

    with open(out_dir / f"seed_{seed}.json", "w") as f:
        json.dump({"base_prompt": base_prompt, "archetype_prompt": archetype_prompt,
                   "seed": seed, "scale": scale, "tau": tau}, f, indent=2)

    print(f"  → {img_path.name}  (archetype: {archetype_prompt})")
    return img_path


# ── Fallback: pipeline call with steered embeds ────────────────────────────────

@torch.inference_mode()
def generate_fallback(
    pipe, base_prompt: str, archetype_prompt: str,
    seed: int, out_dir: Path, device, dtype,
    steps: int, guidance: float, height: int, width: int,
    scale: float, away_prompt: str, away_scale: float, debug: bool,
    prebuilt_embeds=None,
):
    if prebuilt_embeds is not None:
        steered, pooled = prebuilt_embeds
    else:
        _, steered, pooled = build_steered_embeds(
            pipe, base_prompt, archetype_prompt, device, dtype,
            scale, away_prompt, away_scale, debug,
        )

    gen   = torch.Generator(device="cuda" if "cuda" in str(device) else "cpu").manual_seed(seed)
    sig   = inspect.signature(pipe.__call__)
    kwargs = dict(prompt_embeds=steered, height=height, width=width,
                  guidance_scale=guidance, num_inference_steps=steps, generator=gen)
    if "pooled_prompt_embeds" in sig.parameters and pooled is not None:
        kwargs["pooled_prompt_embeds"] = pooled

    img_path = out_dir / f"seed_{seed}.png"
    pipe(**kwargs).images[0].save(img_path)

    with open(out_dir / f"seed_{seed}.json", "w") as f:
        json.dump({"base_prompt": base_prompt, "archetype_prompt": archetype_prompt,
                   "seed": seed, "scale": scale, "tau": "fallback_all_steps"}, f, indent=2)

    print(f"  → {img_path.name}  [fallback]")
    return img_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phased text-embedding steering for FLUX.2 Klein.")
    ap.add_argument("--prompt",       default="a successful person")

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--archetype",   type=str)
    grp.add_argument("--archetypes",  nargs="+")

    ap.add_argument("--pluralistic",  action="store_true",
                    help="Sample archetype uniformly at random per image")
    ap.add_argument("--tau",          type=float, default=0.4,
                    help="Fraction of steps steered exclusively (rest alternates)")
    ap.add_argument("--away-prompt",  type=str,   default=None)
    ap.add_argument("--away-scale",   type=float, default=0.5)
    ap.add_argument("--out",          required=True)
    ap.add_argument("--seeds",        nargs="+", type=int, default=list(range(42, 54)))
    ap.add_argument("--model-id",     default=DEFAULT_MODEL_ID)
    ap.add_argument("--device",       default="cuda:0")
    ap.add_argument("--dtype",        choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--steps",        type=int,   default=50)
    ap.add_argument("--guidance",     type=float, default=4.0)
    ap.add_argument("--height",       type=int,   default=1024)
    ap.add_argument("--width",        type=int,   default=1024)
    ap.add_argument("--scale",        type=float, default=1.5)
    ap.add_argument("--skip-existing",action="store_true")
    ap.add_argument("--cpu-offload",  action="store_true")
    ap.add_argument("--cache-dir",    default=None)
    ap.add_argument("--no-phased",    action="store_true",
                    help="Use steered embeds for all steps (no alternating phase)")
    ap.add_argument("--debug",        action="store_true")
    args = ap.parse_args()

    if args.pluralistic and not args.archetypes:
        ap.error("--pluralistic requires --archetypes")

    apply_cache_dir(args.cache_dir)

    dtype_map   = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device      = torch.device(args.device)
    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {args.model_id}")
    pipe = Flux2KleinPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)

    use_phased = not args.no_phased
    tau_step   = int(args.tau * args.steps)
    print(f"[CONFIG] scale={args.scale} | tau={args.tau} ({tau_step}/{args.steps} steered) | "
          f"phased={'ON' if use_phased else 'OFF'} | seeds={args.seeds}")

    random.seed(42)

    for i, seed in enumerate(args.seeds):
        img_path = out_dir / f"seed_{seed}.png"
        if args.skip_existing and img_path.exists():
            print(f"[SKIP] seed {seed}")
            continue

        archetype = (
            args.archetype if args.archetype
            else random.choice(args.archetypes) if args.pluralistic
            else args.archetypes[i % len(args.archetypes)]
        )
        print(f"\n[GEN] seed {seed} | archetype: '{archetype}'")

        if use_phased:
            _, c_hat_pre, pooled_pre = build_steered_embeds(
                pipe, args.prompt, archetype, device, torch_dtype,
                args.scale, args.away_prompt, args.away_scale, args.debug,
            )
            try:
                generate_phased(
                    pipe=pipe, base_prompt=args.prompt, archetype_prompt=archetype,
                    seed=seed, out_dir=out_dir, device=device, dtype=torch_dtype,
                    steps=args.steps, guidance=args.guidance,
                    height=args.height, width=args.width,
                    scale=args.scale, tau=args.tau,
                    away_prompt=args.away_prompt, away_scale=args.away_scale,
                    debug=args.debug, prebuilt_embeds=(c_hat_pre, pooled_pre),
                )
            except Exception as e:
                print(f"  [WARN] Phased loop failed: {e} — falling back to all-steps steered.")
                generate_fallback(
                    pipe=pipe, base_prompt=args.prompt, archetype_prompt=archetype,
                    seed=seed, out_dir=out_dir, device=device, dtype=torch_dtype,
                    steps=args.steps, guidance=args.guidance,
                    height=args.height, width=args.width,
                    scale=args.scale, away_prompt=args.away_prompt,
                    away_scale=args.away_scale, debug=args.debug,
                    prebuilt_embeds=(c_hat_pre, pooled_pre),
                )
        else:
            generate_fallback(
                pipe=pipe, base_prompt=args.prompt, archetype_prompt=archetype,
                seed=seed, out_dir=out_dir, device=device, dtype=torch_dtype,
                steps=args.steps, guidance=args.guidance,
                height=args.height, width=args.width,
                scale=args.scale, away_prompt=args.away_prompt,
                away_scale=args.away_scale, debug=args.debug,
            )

    print(f"\n[DONE] {len(args.seeds)} images → {out_dir}")


if __name__ == "__main__":
    main()
