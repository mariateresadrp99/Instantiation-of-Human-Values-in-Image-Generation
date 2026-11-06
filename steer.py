"""Social-configuration-guided steering (Sec. 6).

Embedding-space steering injects an auxiliary scene into the post-[EOS] padding positions of the
prompt embedding, leaving the content tokens of the original prompt unchanged:

    c_hat = c_base + λ · M ⊙ (c_aux − c_∅)

c_hat is used for the first τ fraction of denoising steps, then the model reverts to c_base
(FLUX.2 alternates c_hat / c_base after τ). The swap happens through diffusers'
callback_on_step_end, so each pipeline's own sampler and decoder are used unchanged.

Single prompt:
    python steer.py --model flux2 --prompt "a successful person" \
        --scenes "a group of people of different ages celebrating at home" --out results/steered/demo

Full experiment (20 scenes × 2 images + 10 original images per value):
    python steer.py --model flux2 --method embed --aux results/aux_scenes.json \
        --orig_root data/images/flux2 --out data/steered/flux2/embed
"""

import argparse
import json
import shutil

import torch

from utils import image_dir, list_images

DEFAULTS = {  # λ, τ, guidance (Sec. 6)
    "flux2": dict(scale=1.5, tau=0.6, guidance=4.0),
    "sd35": dict(scale=1.0, tau=0.4, guidance=7.0),
    "qwen": dict(scale=1.5, tau=0.4, guidance=6.0),
}


def post_eos_mask(ids, attention, stop_ids, start=0):
    """True from the first EOS/pad token (at or after `start`) onward, and on all padding."""
    ids, real = ids.reshape(-1).cpu(), attention.reshape(-1).cpu().bool()
    pos = torch.arange(len(ids))
    stop = torch.tensor([s for s in stop_ids if s is not None])
    hits = pos[torch.isin(ids, stop) & real & (pos >= start)]
    first = int(hits.min()) if len(hits) else int(pos[real].max()) + 1
    return (pos >= first) | ~real


class Steerer:
    """Model-specific parts: load the pipeline, encode a prompt to a fixed-length embedding,
    and locate the post-EOS positions of the base prompt."""

    def __init__(self, model):
        import diffusers
        self.model = model
        cls, repo = {"flux2": ("Flux2KleinPipeline", "black-forest-labs/FLUX.2-klein-base-9B"),
                     "sd35": ("StableDiffusion3Pipeline", "stabilityai/stable-diffusion-3.5-medium"),
                     "qwen": ("QwenImagePipeline", "Qwen/Qwen-Image")}[model]
        self.pipe = getattr(diffusers, cls).from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def encode(self, prompt):
        p = self.pipe
        if self.model == "flux2":  # Qwen3 hidden states of the chat-templated prompt, length 512
            return p.encode_prompt(prompt, max_sequence_length=512)[0], None
        if self.model == "sd35":  # [CLIP-L ‖ CLIP-G] (77) + T5 (256) along the sequence
            emb, _, pooled, _ = p.encode_prompt(prompt, None, None, do_classifier_free_guidance=False,
                                                max_sequence_length=256)
            return emb, pooled
        # Qwen-Image: diffusers trims padding, so pad to a fixed length and drop the 34 system tokens
        enc = self._qwen_tokens(prompt).to("cuda")
        hidden = p.text_encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                output_hidden_states=True).hidden_states[-1]
        return hidden[:, p.prompt_template_encode_start_idx:], None

    def _qwen_tokens(self, prompt):
        p = self.pipe
        p.tokenizer.padding_side = "right"
        return p.tokenizer(p.prompt_template_encode.format(prompt), padding="max_length", truncation=True,
                           max_length=1024 + p.prompt_template_encode_start_idx, return_tensors="pt")

    def mask(self, prompt):
        p = self.pipe
        if self.model == "flux2":
            text = p.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            enc = p.tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            return post_eos_mask(enc.input_ids, enc.attention_mask, [p.tokenizer.eos_token_id, p.tokenizer.pad_token_id])
        if self.model == "sd35":  # CLIP part untouched; mask starts at 77 + first T5 EOS
            t = p.tokenizer_3
            enc = t(prompt, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
            t5 = post_eos_mask(enc.input_ids, enc.attention_mask, [t.eos_token_id, t.pad_token_id])
            return torch.cat([torch.zeros(77, dtype=torch.bool), t5])
        drop, t = p.prompt_template_encode_start_idx, p.tokenizer  # skip the system prompt's own <|im_end|>
        enc = self._qwen_tokens(prompt)
        return post_eos_mask(enc.input_ids, enc.attention_mask, [t.eos_token_id, t.pad_token_id], start=drop)[drop:]

    @torch.inference_mode()
    def generate(self, prompt, scene, seed, method="embed", scale=None, tau=None, guidance=None, steps=50):
        hp = DEFAULTS[self.model]
        scale, tau, guidance = scale or hp["scale"], tau or hp["tau"], guidance or hp["guidance"]
        kw = dict(num_inference_steps=steps, generator=torch.Generator(self.pipe.device).manual_seed(seed))
        kw["true_cfg_scale" if self.model == "qwen" else "guidance_scale"] = guidance

        if method == "prompt_aug":  # baseline: append the auxiliary scene to the prompt
            extra = dict(negative_prompt=" ") if self.model == "qwen" else {}
            return self.pipe(prompt=f"{prompt}, {scene}", **extra, **kw).images[0]

        (c_base, pooled), (c_aux, _), (c_empty, pooled_empty) = map(self.encode, (prompt, scene, ""))
        m = self.mask(prompt).to(c_base.device, c_base.dtype).view(1, -1, 1)
        c_hat = c_base + scale * m * (c_aux - c_empty)

        n_steered = int(tau * steps)
        alternate = self.model == "flux2"

        def swap(pipe, i, t, tensors):  # chooses the embedding for step i + 1
            nxt = i + 1
            cond = c_hat if nxt < n_steered or (alternate and nxt % 2 == 0) else c_base
            pe = tensors["prompt_embeds"]
            if pe.shape[0] == 2 * cond.shape[0]:  # SD3 keeps [negative; positive] concatenated
                cond = torch.cat([pe[: cond.shape[0]], cond])
            tensors["prompt_embeds"] = cond
            return tensors

        extra = {"sd35": dict(pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=pooled_empty),
                 "qwen": dict(max_sequence_length=1024)}.get(self.model, {})
        return self.pipe(prompt_embeds=c_hat, negative_prompt_embeds=c_empty, callback_on_step_end=swap,
                         callback_on_step_end_tensor_inputs=["prompt_embeds"], **extra, **kw).images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=DEFAULTS, required=True)
    ap.add_argument("--method", choices=["embed", "prompt_aug"], default="embed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", help="single prompt mode")
    ap.add_argument("--scenes", nargs="+", help="auxiliary scenes for --prompt")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--aux", help="aux_scenes.json from generate_auxiliary_scenes.py (experiment mode)")
    ap.add_argument("--orig_root", help="prompt-only images, e.g. data/images/flux2 (experiment mode)")
    ap.add_argument("--scale", type=float)
    ap.add_argument("--tau", type=float)
    args = ap.parse_args()

    from pathlib import Path
    steerer = Steerer(args.model)

    if args.prompt:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        for j, scene in enumerate(args.scenes):
            for seed in args.seeds:
                steerer.generate(args.prompt, scene, seed, args.method, args.scale, args.tau).save(
                    Path(args.out) / f"scene{j:02d}_seed{seed}.png")
        return

    # Experiment: 40 guided images + 10 original images = a 50-image augmented set per value
    for value, info in json.loads(Path(args.aux).read_text())[args.model].items():
        out_dir = image_dir(args.out, info["schwartz_domain"], value, "positive_prompt")
        out_dir.mkdir(parents=True, exist_ok=True)
        originals = list_images(image_dir(args.orig_root, info["schwartz_domain"], value, "positive_prompt"))
        for k, src in enumerate(originals[:10]):
            shutil.copy2(src, out_dir / f"orig_{k:02d}{src.suffix}")
        for j, scene in enumerate(info["scenes"][:20]):
            for r in range(2):
                path = out_dir / f"aux{j:02d}_{r}.png"
                if not path.exists():
                    steerer.generate(info["base_prompt"], scene, 1000 * j + r, args.method,
                                     args.scale, args.tau).save(path)
        print(f"{value}: {len(list_images(out_dir))} images")


if __name__ == "__main__":
    main()
