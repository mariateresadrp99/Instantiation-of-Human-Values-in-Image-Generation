"""Discover global visual prototypes by clustering all VQA descriptions jointly (Sec. 4.2).

Descriptions from all models, values and both polarities are embedded with Qwen3-Embedding-0.6B
and clustered with K-means. K is chosen over {10, 20, ..., 200} as the largest K whose
silhouette score lies within 5% of the best.

    python cluster_prototypes.py --inputs data/vqa/*.json --outdir results/clustering

With --assign, new descriptions (e.g. steered images) are assigned to the existing prototypes:

    python cluster_prototypes.py --inputs data/vqa/steered_flux2_embed.json \
        --outdir results/clustering --assign results/steering/assign_flux2_embed.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import FIELDS, description_text, dominant_tags, embed


def load(paths):
    df = pd.DataFrame([r for p in paths for r in json.loads(Path(p).read_text())])
    return df.drop_duplicates("image_path").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--assign", default=None, help="write assignments of --inputs to existing prototypes here")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.inputs)
    X = embed([description_text(r) for r in df.to_dict("records")])
    cols = ["model", "condition", "schwartz_domain", "value", "prompt_type", "image_path", *FIELDS, "cluster"]

    if args.assign:  # nearest existing centroid
        centroids = np.load(out / "centroids.npy")
        df["cluster"] = (-2 * X @ centroids.T + (centroids ** 2).sum(1)).argmin(1)
        df[cols].to_csv(args.assign, index=False)
        print(f"assigned {len(df)} images → {args.assign}")
        return

    scores, fits = {}, {}
    for k in range(10, 201, 10):
        fits[k] = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(X)
        scores[k] = silhouette_score(X, fits[k].labels_, metric="cosine")
        print(f"K={k:3d} silhouette={scores[k]:.4f}")
    best = max(scores.values())
    k = max(kk for kk, s in scores.items() if s >= best - 0.05 * abs(best))
    print(f"chosen K = {k}")

    df["cluster"] = fits[k].labels_
    df[cols].to_csv(out / "assignments.csv", index=False)
    np.save(out / "centroids.npy", fits[k].cluster_centers_.astype(np.float32))
    tags = {int(c): dominant_tags(g.to_dict("records")) for c, g in df.groupby("cluster")}
    (out / "cluster_tags.json").write_text(json.dumps(tags, indent=1))


if __name__ == "__main__":
    main()
