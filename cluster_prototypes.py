#!/usr/bin/env python3
"""
cluster_prototypes.py
----------------------
Clusters VQA descriptions into global visual prototypes and situations.

Core behaviour:
  1. Load positive + negative VQA JSONs, sample per (model × prompt).
  2. Cluster ONCE on the full pooled set → one shared set of global archetypes.
  3. Write global summary .txt files per scope.
  4. For each polarity independently, write distribution CSVs at three levels:
       domain / value / keyword (prompt_text)

Usage:
    python clustering/cluster_prototypes.py \\
        --input_positive data/vqa/results_gemini_positive.json \\
                         data/vqa/results_flux2_positive.json \\
                         data/vqa/results_qwen_positive.json \\
                         data/vqa/results_sd35_positive.json \\
        --input_negative data/vqa/results_gemini_negative.json \\
                         data/vqa/results_flux2_negative.json \\
                         data/vqa/results_qwen_negative.json \\
                         data/vqa/results_sd35_negative.json \\
        --outdir results/clustering \\
        --sample_per_group 50 \\
        --k_min 3 \\
        --k_max 200 \\
        --sample_col prompt_text
"""

import argparse
import json
import math
import re
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ── Config ─────────────────────────────────────────────────────────────────────

CHANNELS = [
    "people_demographics",
    "people_age",
    "people_appearance",
    "objects",
    "setting",
    "action",
]

SCOPE_TO_CHANNELS = {
    "overall_archetype":    CHANNELS,
    "demo_archetype":       ["people_demographics"],
    "age_archetype":        ["people_age"],
    "appearance_archetype": ["people_appearance"],
    "object_archetype":     ["objects"],
    "setting_archetype":    ["setting"],
    "action_archetype":     ["action"],
}

SCOPE_LABELS = {
    "overall_archetype":    "overall",
    "demo_archetype":       "demo",
    "age_archetype":        "age",
    "appearance_archetype": "appearance",
    "object_archetype":     "objects",
    "setting_archetype":    "setting",
    "action_archetype":     "action",
}

SCOPE_MERGE_THRESHOLDS = {
    "demo_archetype":       0.85,
    "age_archetype":        0.85,
    "overall_archetype":    0.90,
    "setting_archetype":    0.80,
    "action_archetype":     0.80,
    "appearance_archetype": 0.85,
    "object_archetype":     0.80,
}

NONE_LIKE      = {"none", "null", "n/a", "na", "", "none."}
_BASE_EXCLUDE  = {"long_hair", "short_hair", "clothes", "wearing_clothes"}


# ── Token helpers ──────────────────────────────────────────────────────────────

def norm(x: str) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x).strip().lower())


def is_none_like(x) -> bool:
    return norm(x) in NONE_LIKE


def split_items(x) -> list:
    x = norm(x)
    if is_none_like(x):
        return []
    return [p.strip() for p in x.split(",") if p.strip() and not is_none_like(p.strip())]


def parse_demographics(val) -> list:
    if is_none_like(norm(str(val) if val is not None else "")):
        return []
    return re.findall(r"(female|male|non.binary|person)", norm(str(val)))


def row_to_token_doc(row: dict, channels_subset: list, exclude: set) -> str:
    tokens = []
    for ch in channels_subset:
        if ch == "people_demographics":
            for g in parse_demographics(row.get(ch, "")):
                if g not in exclude:
                    tokens.append(g)
        else:
            for it in split_items(row.get(ch, "")):
                it_norm = it.replace(" ", "_")
                if it_norm not in exclude:
                    tokens.append(it_norm)
    return " ".join(tokens)


def row_to_channel_tokens(row: dict, exclude: set) -> dict:
    out = {}
    for ch in CHANNELS:
        if ch == "people_demographics":
            toks = [g for g in parse_demographics(row.get(ch, "")) if g not in exclude]
        else:
            toks = [
                it.replace(" ", "_")
                for it in split_items(row.get(ch, ""))
                if it.replace(" ", "_") not in exclude
            ]
        out[ch] = toks
    return out


# ── K selection ────────────────────────────────────────────────────────────────

def choose_k_by_silhouette(X, k_min: int, k_max: int, seed: int):
    n = X.shape[0]
    k_max = min(k_max, n - 1)
    k_min = max(2, min(k_min, k_max))
    if n < 3 or k_max < 2:
        return 1, {}

    scores = {}
    step   = 5 if (k_max - k_min) > 50 else 1
    k_vals = list(range(k_min, k_max + 1, step))
    if k_max not in k_vals:
        k_vals.append(k_max)

    for k in k_vals:
        km     = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto", batch_size=2048)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        scores[k] = float(
            silhouette_score(X, labels, metric="cosine", sample_size=2500, random_state=seed)
        )

    if not scores:
        return 1, {}

    best_score = scores[max(scores, key=scores.get)]
    qualified  = [k for k, s in scores.items() if s >= best_score * 0.95]
    return max(qualified), scores


# ── Metrics ────────────────────────────────────────────────────────────────────

def distribution_metrics_from_labels(labels):
    labels = np.asarray(labels)[np.asarray(labels) >= 0]
    if len(labels) == 0:
        return pd.Series(dtype=int), float("nan"), 0.0, 0
    counts    = pd.Series(labels).value_counts().sort_index()
    n         = int(counts.sum())
    p         = (counts / n).values.astype(float)
    K_present = len(p)
    asd       = float(p.max()) if K_present > 0 else float("nan")
    H_raw     = 0.0 if K_present <= 1 else -float(np.sum([pi * math.log(pi) for pi in p if pi > 0]))
    return counts, asd, H_raw, K_present


def top_tokens_from_counter(ctr: Counter, top_n: int) -> list:
    if not ctr:
        return []
    return [t for t, _ in sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]


def pct_bar(pct: float, width: int = 30) -> str:
    filled = int(round(pct * width))
    return "█" * filled + " " * (width - filled)


def maybe_sample(df: pd.DataFrame, n, seed: int) -> pd.DataFrame:
    if n is not None and len(df) > n:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


# ── Cosine merging ─────────────────────────────────────────────────────────────

def merge_similar_clusters(X, labels, threshold: float = 0.70):
    while True:
        unique = sorted(set(labels))
        if len(unique) <= 1:
            break
        centroids  = np.array([np.mean(X[labels == c], axis=0) for c in unique])
        sim_matrix = cosine_similarity(centroids)
        np.fill_diagonal(sim_matrix, -1)
        if np.max(sim_matrix) < threshold:
            break
        i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
        labels[labels == unique[j]] = unique[i]

    final = np.zeros_like(labels)
    for new_idx, old_c in enumerate(sorted(set(labels))):
        final[labels == old_c] = new_idx
    return final, len(set(final))


def cluster_docs(docs, model, k_min, k_max, seed, merge_threshold):
    clean_docs  = [d if isinstance(d, str) else "" for d in docs]
    X_emb       = np.asarray(
        model.encode(clean_docs, batch_size=64, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )

    nan_rows    = np.isnan(X_emb).any(axis=1)
    valid_idx   = [i for i, d in enumerate(clean_docs) if d.strip() and not nan_rows[i]]
    labels      = np.full(len(docs), -1, dtype=int)

    if len(set(clean_docs[i] for i in valid_idx)) < 3 or len(valid_idx) < 5:
        if valid_idx:
            labels[valid_idx] = 0
        counts, asd, H, K = distribution_metrics_from_labels(labels)
        return labels, counts, asd, H, K, float("nan"), 1, X_emb

    X_valid  = X_emb[valid_idx]
    k_max_eff = min(k_max, len(set(clean_docs[i] for i in valid_idx)) - 1, len(valid_idx) - 1)
    k_min_eff = max(2, min(k_min, k_max_eff))

    chosen_k, _ = choose_k_by_silhouette(X_valid, k_min_eff, k_max_eff, seed)

    if chosen_k <= 1:
        labels[valid_idx] = 0
        sil = float("nan")
    else:
        km           = KMeans(n_clusters=chosen_k, random_state=seed, n_init="auto")
        valid_labels = km.fit_predict(X_valid)
        original_k   = chosen_k
        valid_labels, chosen_k = merge_similar_clusters(X_valid, valid_labels, merge_threshold)

        if original_k != chosen_k:
            print(f"  [MERGE] {original_k} → {chosen_k} clusters (threshold={merge_threshold})")

        labels[valid_idx] = valid_labels
        sil = (
            silhouette_score(X_valid, valid_labels, metric="cosine",
                             sample_size=min(2500, len(X_valid)), random_state=seed)
            if chosen_k > 1 else float("nan")
        )

    counts, asd, H, K = distribution_metrics_from_labels(labels)
    return labels, counts, asd, H, K, sil, chosen_k, X_emb


# ── Summary writer ─────────────────────────────────────────────────────────────

def write_global_summary(df, labels, counts, chosen_k, scope, out_summaries, top_n_many, exclude):
    records        = df.to_dict(orient="records")
    ch_tokens      = [row_to_channel_tokens(r, exclude) for r in records]
    cluster_ctr    = {c: {ch: Counter() for ch in CHANNELS} for c in range(chosen_k)}
    total_valid    = int(np.sum(labels >= 0))

    for i, c in enumerate(labels.tolist()):
        if c == -1:
            continue
        for ch in CHANNELS:
            cluster_ctr[c][ch].update(ch_tokens[i].get(ch, []))

    lines = [
        f"GLOBAL ARCHETYPES / {scope}",
        f"(N={len(records)}, valid={total_valid}, K={chosen_k})",
    ]

    for c in range(chosen_k):
        cnt = counts.get(c, 0)
        pct = float(cnt) / float(total_valid) if total_valid > 0 else 0
        bar = pct_bar(pct, 30)

        def fmt1(ch, _c=c):
            tok = top_tokens_from_counter(cluster_ctr[_c][ch], 1)
            if not tok:
                return ""
            t = tok[0].replace("_", " ")
            if ch == "people_demographics":
                m = re.search(r"(female|male|non.binary)", t)
                return m.group(1) if m else ""
            return t

        def fmtN(ch, _c=c):
            ctr = cluster_ctr[_c][ch]
            if not ctr:
                return ""
            top = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n_many]
            return ", ".join(f"{t.replace('_', ' ')} ({n})" for t, n in top)

        lines.append(
            f"[{bar}] {pct*100:5.1f}% Cluster {c} | "
            f"demo={fmt1('people_demographics')} | age={fmt1('people_age')} | "
            f"action={fmt1('action')} | appearance={fmtN('people_appearance')} | "
            f"objects={fmtN('objects')}"
        )

    (out_summaries / f"summary__GLOBAL__{scope}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ── Distribution writers ───────────────────────────────────────────────────────

def _get_dist(labels_subset, chosen_k):
    valid   = labels_subset[labels_subset >= 0]
    n_valid = len(valid)
    if n_valid == 0:
        return np.zeros(chosen_k), 0
    counts = np.array([np.sum(valid == c) for c in range(chosen_k)], dtype=float)
    return counts / n_valid, n_valid


def write_distribution_domain(df_pol, scope_info, pol_dir, polarity_label):
    out_dir = pol_dir / "domain"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scope, info in scope_info.items():
        rows = []
        for (model, domain), df_sub in df_pol.groupby(["model", "schwartz_domain"]):
            dist, n_valid = _get_dist(df_sub[info["col"]].values, info["chosen_k"])
            row = {"Model": model, "Schwartz_domain": domain,
                   "Total_Images": len(df_sub), "Valid_Images": n_valid}
            for c in range(info["chosen_k"]):
                row[f"Cluster_{c}_%"] = round(dist[c] * 100, 2)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / f"domain_matrix__{scope}.csv", index=False)


def write_distribution_value(df_pol, scope_info, pol_dir, polarity_label):
    out_dir = pol_dir / "value"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scope, info in scope_info.items():
        rows = []
        for (model, value), df_sub in df_pol.groupby(["model", "value"]):
            dist, n_valid = _get_dist(df_sub[info["col"]].values, info["chosen_k"])
            row = {"Model": model, "Schwartz_value": value,
                   "Total_Images": len(df_sub), "Valid_Images": n_valid}
            for c in range(info["chosen_k"]):
                row[f"Cluster_{c}_%"] = round(dist[c] * 100, 2)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / f"value_matrix__{scope}.csv", index=False)


def write_distribution_keyword(df_pol, scope_info, pol_dir, polarity_label):
    out_kw = pol_dir / "prompt_text"
    for sv, df_sv in df_pol.groupby("value"):
        val_dir = out_kw / str(sv)
        val_dir.mkdir(parents=True, exist_ok=True)
        for scope, info in scope_info.items():
            rows = []
            for (model, kw), df_sub in df_sv.groupby(["model", "prompt_text"]):
                dist, n_valid = _get_dist(df_sub[info["col"]].values, info["chosen_k"])
                row = {"Model": model, "Keyword": kw,
                       "Total_Images": len(df_sub), "Valid_Images": n_valid}
                for c in range(info["chosen_k"]):
                    row[f"Cluster_{c}_%"] = round(dist[c] * 100, 2)
                rows.append(row)
            pd.DataFrame(rows).to_csv(
                val_dir / f"keyword_matrix__{sv}__{scope}.csv", index=False
            )


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(input_paths, status_filter: str, polarity_tag: str) -> list:
    rows = []
    for inp in input_paths:
        path   = Path(inp)
        data   = json.loads(path.read_text(encoding="utf-8"))
        inferred_model = (
            path.stem
            .replace("results_", "")
            .replace("_positive", "")
            .replace("_negative", "")
        )
        for r in data:
            if norm(r.get("status")) == norm(status_filter):
                if "model" not in r:
                    r["model"] = r.get("model_name", inferred_model)
                r["_polarity"] = polarity_tag
                rows.append(r)
    return rows


def build_exclude_tokens(df: pd.DataFrame, base_exclude: set, min_freq: int = 5) -> set:
    all_tokens = []
    for r in df.to_dict(orient="records"):
        for ch in CHANNELS:
            if ch == "people_demographics":
                all_tokens.extend(parse_demographics(r.get(ch, "")))
            else:
                for it in split_items(r.get(ch, "")):
                    all_tokens.append(it.replace(" ", "_"))
    token_counts = Counter(all_tokens)
    rare = {tok for tok, cnt in token_counts.items() if cnt < min_freq}
    print(f"  Ignored {len(rare)} rare tokens (freq < {min_freq}). Kept {len(token_counts) - len(rare)}.")
    return base_exclude | rare


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Cluster VQA descriptions into global visual prototypes.")
    ap.add_argument("--input_positive",  nargs="+", required=True)
    ap.add_argument("--input_negative",  nargs="+", required=True)
    ap.add_argument("--outdir",          required=True)
    ap.add_argument("--status_filter",   default="ok")
    ap.add_argument("--k_min",           type=int,   default=5)
    ap.add_argument("--k_max",           type=int,   default=200)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--embed_model",     default="Alibaba-NLP/gte-base-en-v1.5")
    ap.add_argument("--top_n_many",      type=int,   default=10)
    ap.add_argument("--sample_per_group",type=int,   default=50)
    ap.add_argument("--merge_threshold", type=float, default=0.85)
    ap.add_argument("--sample_col",      default="prompt_text",
                    help="Column to group by for sampling.")
    args = ap.parse_args()

    sample_n = args.sample_per_group if args.sample_per_group > 0 else None
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[LOAD] Positive ...")
    pos_rows = load_data(args.input_positive, args.status_filter, "positive")
    print(f"  {len(pos_rows)} rows")

    print("[LOAD] Negative ...")
    neg_rows = load_data(args.input_negative, args.status_filter, "negative")
    print(f"  {len(neg_rows)} rows")

    df_all = pd.DataFrame(pos_rows + neg_rows)

    def sample_df(df):
        parts = []
        for _, ddf in df.groupby(["model", args.sample_col]):
            parts.append(maybe_sample(ddf, sample_n, args.seed))
        return pd.concat(parts, ignore_index=True)

    df_pos  = sample_df(df_all[df_all["_polarity"] == "positive"].copy())
    df_neg  = sample_df(df_all[df_all["_polarity"] == "negative"].copy())
    df_pool = pd.concat([df_pos, df_neg], ignore_index=True)
    print(f"[INFO] Pooled: {len(df_pool)} images")

    exclude   = build_exclude_tokens(df_pool, set(_BASE_EXCLUDE))
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Embedding model: {args.embed_model} on {device}")
    emb_model = SentenceTransformer(args.embed_model, trust_remote_code=True, device=device)

    out_summaries = outdir / "summaries"
    out_summaries.mkdir(parents=True, exist_ok=True)

    scope_info  = {}
    all_metrics = []
    records     = df_pool.to_dict(orient="records")

    for scope in SCOPE_TO_CHANNELS:
        print(f"\n[CLUSTER] {scope} ...")
        docs  = [row_to_token_doc(r, SCOPE_TO_CHANNELS[scope], exclude) for r in records]
        merge = SCOPE_MERGE_THRESHOLDS.get(scope, args.merge_threshold)

        labels, counts, asd, H_raw, K_present, sil, chosen_k, X_emb = cluster_docs(
            docs, emb_model, args.k_min, args.k_max, args.seed, merge
        )

        scope_col = f"global_cluster_{scope}"
        df_pool[scope_col] = labels
        scope_info[scope]  = {"chosen_k": chosen_k, "col": scope_col}

        valid_n = int(np.sum(labels >= 0))
        sil_str = f" sil={sil:.4f}" if not np.isnan(sil) else ""
        print(f"  valid={valid_n}/{len(df_pool)} K={chosen_k}{sil_str}")

        write_global_summary(df_pool, labels, counts, chosen_k, scope,
                             out_summaries, args.top_n_many, exclude)

        for polarity, df_sub in [("positive", df_pos), ("negative", df_neg)]:
            pool_idx = df_pool[df_pool["_polarity"] == polarity].index
            for target_col in ["schwartz_domain", "value"]:
                if target_col not in df_pool.columns:
                    continue
                for (model_name, tval), idx_sub in df_pool.loc[pool_idx].groupby(
                    ["model", target_col]
                ).groups.items():
                    dom_labels = df_pool.loc[idx_sub, scope_col].values
                    _, d_asd, d_H, _ = distribution_metrics_from_labels(dom_labels)
                    all_metrics.append({
                        "polarity":    polarity,
                        "model":       model_name,
                        "target_type": target_col,
                        "target_val":  tval,
                        "scope":       scope,
                        "scope_label": SCOPE_LABELS.get(scope, scope),
                        "n_images":    len(idx_sub),
                        "n_valid":     int(np.sum(dom_labels >= 0)),
                        "global_k":    chosen_k,
                        "asd":         float(d_asd),
                        "H_raw":       float(d_H),
                    })

    for scope in SCOPE_TO_CHANNELS:
        col = scope_info[scope]["col"]
        df_pos[col] = df_pool.loc[df_pool["_polarity"] == "positive", col].values
        df_neg[col] = df_pool.loc[df_pool["_polarity"] == "negative", col].values

    for polarity_label, df_pol in [("positive", df_pos), ("negative", df_neg)]:
        pol_dir = outdir / polarity_label
        pol_dir.mkdir(parents=True, exist_ok=True)
        write_distribution_domain(df_pol, scope_info, pol_dir, polarity_label)
        write_distribution_value(df_pol, scope_info, pol_dir, polarity_label)
        write_distribution_keyword(df_pol, scope_info, pol_dir, polarity_label)

    pd.DataFrame(all_metrics).to_csv(outdir / "overview_metrics.csv", index=False)
    print(f"\n[DONE] Results written to {outdir}/")


if __name__ == "__main__":
    main()
