"""Label each visual prototype with an LLM at two levels (Sec. 4.2): a fine-grained visual
prototype label and a broader situation label.

    python label_prototypes.py --clusters results/clustering/cluster_tags.json \
        --output results/clustering/prototype_labels.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from utils import llm, parse_json

PROMPT = """You are labeling a cluster of images based on VQA-generated tags.
Each cluster contains images that are visually and semantically similar.

You will be given the dominant tags for a cluster, including
demographic, setting, action, appearance, and objects. Counts in
parentheses indicate how often each tag appears in the cluster.
Use high-frequency tags as stronger evidence than low-frequency tags.

--------------------------------------------------
LABEL DEFINITIONS
--------------------------------------------------

visual_prototype_label
  A concrete description of the recurring visual scene.
  Preserve visible details such as subject demographics, clothing,
  action, objects, and setting when they are central to the cluster.
  Use a short noun phrase, no more than 15 words.

    Example input:
      demographic=adult male; setting=office; action=sitting;
      appearance=suit (1233), tie (879), shirt (605), pants (206),
      white shirt (115), watch (110), glasses (105);
      objects=desk (730), laptop (376), books (277), chair (272),
      pen (193), lamp (158)

    Example label:
      adult man in a suit sitting at a desk in an office

situation_label
  A broader situation type that captures the underlying activity
  or context. Abstract away from demographic differences and minor
  visual details where possible.
  Use a short phrase, no more than 5 words.

    Example label:
      office work

--------------------------------------------------
OUTPUT FORMAT (JSON ONLY)
--------------------------------------------------
{
  "visual_prototype_label": "adult man in a suit sitting at a desk in an office",
  "situation_label": "office work"
}

--------------------------------------------------
INPUT
--------------------------------------------------

Dominant tags:
{dominant_tags}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--llm_url", default="http://localhost:8000/v1")
    args = ap.parse_args()

    rows = []
    for cluster, tags in json.loads(Path(args.clusters).read_text()).items():
        labels = parse_json(llm(PROMPT.replace("{dominant_tags}", tags), base_url=args.llm_url))
        rows.append({"cluster": int(cluster), "visual_prototype_label": labels["visual_prototype_label"],
                     "situation": labels["situation_label"].strip().lower().rstrip(".")})
        print(cluster, "|", rows[-1]["visual_prototype_label"], "|", rows[-1]["situation"])
    pd.DataFrame(rows).sort_values("cluster").to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
