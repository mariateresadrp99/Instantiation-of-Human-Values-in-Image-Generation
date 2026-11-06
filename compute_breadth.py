"""Instantiation breadth (Sec. 4.3): N_eff^vp over visual prototypes and N_eff^sit over
situations, for every model × value × prompt type.

    python compute_breadth.py --assignments results/clustering/assignments.csv \
        --labels results/clustering/prototype_labels.csv --output results/breadth.csv
"""

import argparse

import pandas as pd

from utils import neff

GROUP = ["model", "condition", "schwartz_domain", "value", "prompt_type"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", nargs="+", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.concat(map(pd.read_csv, args.assignments)).merge(pd.read_csv(args.labels), on="cluster")
    breadth = df.groupby(GROUP).agg(n_images=("cluster", "size"),
                                    neff_vp=("cluster", neff),
                                    neff_sit=("situation", neff)).reset_index()
    breadth.to_csv(args.output, index=False)

    print(f"{df['cluster'].nunique()} visual prototypes, {df['situation'].nunique()} situations\n")
    for by in (["condition", "prompt_type"], ["condition", "model"], ["condition", "schwartz_domain"]):
        print(breadth.groupby(by)[["neff_vp", "neff_sit"]].mean().round(2), "\n")


if __name__ == "__main__":
    main()
