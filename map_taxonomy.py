"""Map visual prototypes onto the social-configuration taxonomy (Sec. 5).

For each axis (Setting, Sociality, Mode of Engagement, Social Register), each prototype is
assigned to the category with the highest cosine similarity to the category description,
by majority vote across three text encoders. Outputs the per-prototype categories and the
share of images in each category per Schwartz domain.

    python map_taxonomy.py --labels results/clustering/prototype_labels.csv \
        --assignments results/clustering/assignments.csv --outdir results/taxonomy
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from utils import embed

ENCODERS = ["Qwen/Qwen3-Embedding-0.6B", "google/embeddinggemma-300m", "Alibaba-NLP/gte-large-en-v1.5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--assignments", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--taxonomy", default="taxonomy_definitions.json")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    taxonomy = {k: v for k, v in json.loads(Path(args.taxonomy).read_text()).items() if not k.startswith("_")}
    protos = pd.read_csv(args.labels)
    texts = protos["visual_prototype_label"].tolist()

    for axis, spec in taxonomy.items():
        names, descriptions = list(spec["categories"]), list(spec["categories"].values())
        votes = [[names[i] for i in (embed(texts, enc) @ embed(descriptions, enc).T).argmax(1)] for enc in ENCODERS]
        # majority vote; with three different answers, keep the first encoder's
        protos[axis] = [Counter(v).most_common(1)[0][0] if len(set(v)) < 3 else v[0] for v in zip(*votes)]
    protos.to_csv(out / "prototype_taxonomy.csv", index=False)

    df = pd.concat(map(pd.read_csv, args.assignments)).merge(protos, on="cluster")
    for axis in taxonomy:
        table = pd.crosstab([df["condition"], df["prompt_type"], df["schwartz_domain"]], df[axis], normalize="index")
        (table * 100).round(1).to_csv(out / f"{axis.replace(' ', '_')}_by_domain.csv")
        print(f"\n{axis} (% of images)\n", (pd.crosstab(df["prompt_type"], df[axis], normalize="index") * 100).round(1))


if __name__ == "__main__":
    main()
