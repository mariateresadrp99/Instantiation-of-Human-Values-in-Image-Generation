#!/usr/bin/env python3
"""
generate_images_gemini.py
--------------------------
Bulk image generator for google/gemini-2.5-flash-image via the Replicate API.

Usage:
    export REPLICATE_API_TOKEN=<your_token>

    python generation/generate_images_gemini.py \\
        --csv prompts.csv \\
        --out data/images/gemini \\
        --images-per-prompt 50 \\
        --which both
"""

import os
import re
import csv
import argparse
import time
import random
from pathlib import Path
from typing import List, Dict, Optional

import requests
import replicate
from replicate.exceptions import ModelError, ReplicateError


DEFAULT_MODEL = "google/gemini-2.5-flash-image"
SENSITIVE_MARKERS = ("E005", "flagged as sensitive")


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
            raise ValueError(f"CSV missing columns: {missing}")
        for r in reader:
            if r.get("value") and r.get("schwartz_domain"):
                rows.append(r)
    return rows


def count_existing(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.jpg")) + sum(1 for _ in folder.glob("*.png"))


def download_file(url: str, out_path: Path, timeout: int = 180):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)


def is_sensitive_error(msg: str) -> bool:
    lower = msg.lower()
    return any(m.lower() in lower for m in SENSITIVE_MARKERS)


# ── Replicate runner ───────────────────────────────────────────────────────────

def run_gemini_flash(
    client: replicate.Client,
    model_id: str,
    prompt: str,
    aspect_ratio: str = "1:1",
) -> str:
    pred = client.predictions.create(
        model=model_id,
        input={"prompt": prompt, "aspect_ratio": aspect_ratio},
    )
    pred.wait()

    if pred.status != "succeeded":
        raise RuntimeError(
            f"Prediction failed: status={pred.status} error={pred.error} logs={pred.logs}"
        )

    output = pred.output
    if isinstance(output, list):
        if not output:
            raise RuntimeError("Empty output list from Replicate.")
        out0 = output[0]
        return out0.url if hasattr(out0, "url") else str(out0)
    return output.url if hasattr(output, "url") else str(output)


# ── Resumable generation ───────────────────────────────────────────────────────

def generate_resumable(
    client: replicate.Client,
    model_id: str,
    prompt: str,
    out_dir: Path,
    n_images: int,
    base_name: str,
    *,
    max_retries: int = 6,
    retry_base_sleep: float = 0.75,
    failed_log_path: Optional[Path] = None,
    throttle_after_success_s: float = 0.0,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = count_existing(out_dir)
    if existing >= n_images:
        print(f"[SKIP] {out_dir} already has {existing}/{n_images} images.")
        return

    sensitive_log_path: Optional[Path] = (
        failed_log_path.parent / "_sensitive_prompts.tsv" if failed_log_path else None
    )

    def log_sensitive(filename: str, reason: str, err: BaseException) -> None:
        if sensitive_log_path:
            sensitive_log_path.parent.mkdir(parents=True, exist_ok=True)
            with sensitive_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{out_dir}\t{filename}\t{reason}\t{err}\tPROMPT={prompt!r}\n")

    prompt_blocked = False

    for i in range(n_images):
        if prompt_blocked:
            break

        out_path = out_dir / f"{base_name}_{i:04d}.jpg"
        if out_path.exists():
            continue

        last_err: Optional[BaseException] = None
        generic_fail_count = 0

        for attempt in range(1, max_retries + 1):
            try:
                url = run_gemini_flash(client, model_id, prompt)
                download_file(url, out_path)
                print(f"[OK] {out_path}")
                last_err = None
                if throttle_after_success_s > 0:
                    time.sleep(throttle_after_success_s)
                break

            except (ModelError, ReplicateError, RuntimeError) as e:
                last_err = e
                msg = str(e)
                print(f"[ERROR] {out_path.name} attempt {attempt}/{max_retries}: {msg}")

                if is_sensitive_error(msg):
                    print(f"[SENSITIVE] E005 block — abandoning remaining images for this prompt.")
                    log_sensitive(out_path.name, "E005", e)
                    last_err = None
                    prompt_blocked = True
                    break

                if "Failed to generate image" in msg:
                    generic_fail_count += 1
                    if generic_fail_count >= 2:
                        print(f"[SENSITIVE?] {generic_fail_count}x generic failure — likely policy block.")
                        log_sensitive(out_path.name, "GENERIC_FAIL", e)
                        last_err = None
                        prompt_blocked = True
                        break

                sleep_s = retry_base_sleep * (2 ** (attempt - 1))
                if "status: 429" in msg or "Request was throttled" in msg or "rate limit" in msg.lower():
                    sleep_s = max(sleep_s, 1.25) + random.uniform(0.0, 0.6)
                else:
                    sleep_s += random.uniform(0.0, 0.4)
                time.sleep(sleep_s)

            except Exception as e:
                last_err = e
                print(f"[UNEXPECTED] {out_path.name} attempt {attempt}/{max_retries}: {e}")
                time.sleep(retry_base_sleep * (2 ** (attempt - 1)) + random.uniform(0.0, 0.6))

        if last_err is not None:
            print(f"[FAIL] Skipping {out_path.name} after {max_retries} retries.")
            if failed_log_path:
                failed_log_path.parent.mkdir(parents=True, exist_ok=True)
                with failed_log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"{out_dir}\t{out_path.name}\t{type(last_err).__name__}\t{last_err}"
                        f"\tPROMPT={prompt!r}\n"
                    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Bulk image generator using google/gemini-2.5-flash-image via Replicate."
    )
    ap.add_argument("--csv",                required=True,  help="Path to prompts CSV")
    ap.add_argument("--out",                required=True,  help="Output root directory")
    ap.add_argument("--model-id",           default=DEFAULT_MODEL)
    ap.add_argument("--images-per-prompt",  type=int,   default=50)
    ap.add_argument("--which",              choices=["pos", "neg", "both"], default="both")
    ap.add_argument("--max-retries",        type=int,   default=6)
    ap.add_argument("--retry-base-sleep",   type=float, default=0.75)
    ap.add_argument("--throttle",           type=float, default=0.0,
                    help="Fixed sleep (s) after each successful image to reduce 429s.")
    args = ap.parse_args()

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("Set REPLICATE_API_TOKEN: export REPLICATE_API_TOKEN=<your_token>")

    client   = replicate.Client(api_token=token)
    rows     = read_schwartz_csv(Path(args.csv))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    failed_log = out_root / "_failed_prompts.tsv"

    print(f"[INFO] {len(rows)} prompts loaded.")

    for row_idx, r in enumerate(rows):
        value   = safe_slug(r.get("value"))
        domain  = safe_slug(r.get("schwartz_domain"))
        pos_p   = (r.get("positive_prompt") or "").strip()
        neg_p   = (r.get("negative_prompt") or "").strip()
        base_dir = out_root / domain / value
        base_name = f"row{row_idx:05d}"

        if args.which in ("pos", "both") and pos_p:
            print(f"[ROW {row_idx}] POS {domain}/{value}")
            generate_resumable(
                client=client, model_id=args.model_id, prompt=pos_p,
                out_dir=base_dir / "positive_prompt", n_images=args.images_per_prompt,
                base_name=base_name + "_pos", max_retries=args.max_retries,
                retry_base_sleep=args.retry_base_sleep, failed_log_path=failed_log,
                throttle_after_success_s=args.throttle,
            )

        if args.which in ("neg", "both") and neg_p:
            print(f"[ROW {row_idx}] NEG {domain}/{value}")
            generate_resumable(
                client=client, model_id=args.model_id, prompt=neg_p,
                out_dir=base_dir / "negative_prompt", n_images=args.images_per_prompt,
                base_name=base_name + "_neg", max_retries=args.max_retries,
                retry_base_sleep=args.retry_base_sleep, failed_log_path=failed_log,
                throttle_after_success_s=args.throttle,
            )

    print("[DONE]")


if __name__ == "__main__":
    main()
