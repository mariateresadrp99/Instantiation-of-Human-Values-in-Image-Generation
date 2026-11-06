"""Shared helpers: prompt table, folders, VQA text, embeddings, LLM calls, and N_eff (Eq. 1)."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

FIELDS = ["people_demographics", "people_appearance", "objects", "setting", "action"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"


# ── Prompts and folders ────────────────────────────────────────────────────────

def read_prompts(path="prompts.csv"):
    with open(path, encoding="utf-8") as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


def slug(s):
    return re.sub(r"\s+", "_", re.sub(r"[^\w\-. ]+", "_", s.strip()))


def image_dir(root, domain, value, prompt_type):
    """data/images/<model>/<domain>/<value>/<positive_prompt|negative_prompt>/"""
    return Path(root) / slug(domain) / slug(value) / prompt_type


def list_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS) if folder.is_dir() else []


# ── VQA descriptions ───────────────────────────────────────────────────────────

def clean(value):
    if isinstance(value, (list, tuple)):
        value = ", ".join(map(str, value))
    value = re.sub(r"\s+", " ", str(value or "none").strip().lower())
    return "none" if value in {"", "null", "n/a", "none."} else value


def tokens(value):
    return [t.strip(" .").replace("_", " ") for t in clean(value).split(",") if t.strip(" .") not in {"", "none"}]


def description_text(rec):
    """Concatenate the five VQA answers into one structured description (Sec. 4.1)."""
    return " ".join(f"{f.replace('_', ' ')}: {clean(rec.get(f))}." for f in FIELDS)


def dominant_tags(records, top_n=10):
    """Most frequent tags per field in a cluster, rendered as in App. D.2."""
    ctr = {f: Counter(t for r in records for t in tokens(r.get(f))) for f in FIELDS}
    top = lambda f, n: ctr[f].most_common(n)  # noqa: E731
    first = lambda f: top(f, 1)[0][0] if ctr[f] else "none"  # noqa: E731
    many = lambda f: ", ".join(f"{t} ({c})" for t, c in top(f, top_n)) or "none"  # noqa: E731
    return (f"demographic={first('people_demographics')}; setting={first('setting')}; "
            f"action={first('action')}; appearance={many('people_appearance')}; objects={many('objects')}")


# ── Models ─────────────────────────────────────────────────────────────────────

_encoders = {}


def embed(texts, model=EMBED_MODEL):
    """L2-normalised sentence embeddings."""
    if model not in _encoders:
        from sentence_transformers import SentenceTransformer
        _encoders[model] = SentenceTransformer(model, trust_remote_code=True)
    return np.asarray(_encoders[model].encode(list(texts), batch_size=64, normalize_embeddings=True), np.float32)


def llm(prompt, base_url="http://localhost:8000/v1", model=LLM_MODEL, temperature=0.0):
    """One chat call to an OpenAI-compatible server (e.g. `vllm serve Qwen/Qwen2.5-72B-Instruct`)."""
    from openai import OpenAI
    out = OpenAI(base_url=base_url, api_key="EMPTY").chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=1024)
    return out.choices[0].message.content


def parse_json(text):
    return json.loads(re.search(r"\{.*\}", text, re.DOTALL).group())


# ── Breadth ────────────────────────────────────────────────────────────────────

def neff(labels):
    """Inverse Simpson index N_eff = 1 / sum_c pi_c^2 (Eq. 1)."""
    counts = np.array(list(Counter(labels).values()), dtype=float)
    p = counts / counts.sum()
    return float(1.0 / (p ** 2).sum())
