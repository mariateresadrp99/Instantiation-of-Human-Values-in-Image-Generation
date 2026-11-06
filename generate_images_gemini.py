"""Generate N images per value prompt with Gemini 2.5 Flash Image via Replicate (Sec. 3).

    export REPLICATE_API_TOKEN=<your_token>
    python generate_images_gemini.py --out data/images/gemini
"""

import argparse
import time

import replicate
import requests

from utils import image_dir, read_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="prompts.csv")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    for row in read_prompts(args.csv):
        for prompt_type in ["positive_prompt", "negative_prompt"]:
            out_dir = image_dir(args.out, row["schwartz_domain"], row["value"], prompt_type)
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(args.n):
                path = out_dir / f"{i:04d}.png"
                for attempt in range(args.retries):
                    if path.exists():
                        break
                    try:
                        out = replicate.run("google/gemini-2.5-flash-image",
                                            input={"prompt": row[prompt_type], "aspect_ratio": "1:1",
                                                   "output_format": "png"})
                        url = out.url if hasattr(out, "url") else str(out[0] if isinstance(out, list) else out)
                        path.write_bytes(requests.get(url, timeout=180).content)
                    except Exception as e:  # safety refusals, rate limits, timeouts
                        print(f"  {path.name} attempt {attempt + 1}: {e}")
                        time.sleep(2 ** attempt)
            print(f"{row['value']}/{prompt_type}: {len(list(out_dir.glob('*.png')))} images")


if __name__ == "__main__":
    main()
