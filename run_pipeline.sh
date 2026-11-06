#!/usr/bin/env bash
# Full pipeline. Needs a GPU, REPLICATE_API_TOKEN for Gemini, and an LLM server for labeling:
#   vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4 &
set -euo pipefail

# 1. Generate 50 images per prompt with each model
for M in flux2 qwen sd35; do python generate_images.py --model $M --out data/images/$M; done
python generate_images_gemini.py --out data/images/gemini

# 2. Describe each image with five VQA questions
for M in flux2 qwen sd35 gemini; do
  python extract_vqa_descriptions.py --img_root data/images/$M --model_name $M --output data/vqa/$M.json
done

# 3. Cluster all descriptions into visual prototypes and label them
python cluster_prototypes.py --inputs data/vqa/{flux2,qwen,sd35,gemini}.json --outdir results/clustering
python label_prototypes.py --clusters results/clustering/cluster_tags.json --output results/clustering/prototype_labels.csv

# 4. Instantiation breadth and social-configuration taxonomy
python compute_breadth.py --assignments results/clustering/assignments.csv \
    --labels results/clustering/prototype_labels.csv --output results/breadth.csv
python map_taxonomy.py --labels results/clustering/prototype_labels.csv \
    --assignments results/clustering/assignments.csv --outdir results/taxonomy

# 5. Steering: auxiliary scenes → guided images → breadth of the augmented sets
python generate_auxiliary_scenes.py --breadth results/breadth.csv --assignments results/clustering/assignments.csv \
    --labels results/clustering/prototype_labels.csv --output results/aux_scenes.json
ASSIGN=()
for M in flux2 qwen sd35; do for METHOD in prompt_aug embed; do
  python steer.py --model $M --method $METHOD --aux results/aux_scenes.json \
      --orig_root data/images/$M --out data/steered/$M/$METHOD
  python extract_vqa_descriptions.py --img_root data/steered/$M/$METHOD --model_name $M --condition $METHOD \
      --prompt_types positive_prompt --output data/vqa/steered_${M}_${METHOD}.json
  python cluster_prototypes.py --inputs data/vqa/steered_${M}_${METHOD}.json --outdir results/clustering \
      --assign results/clustering/steered_${M}_${METHOD}.csv
  ASSIGN+=(results/clustering/steered_${M}_${METHOD}.csv)
done; done
python compute_breadth.py --assignments "${ASSIGN[@]}" --labels results/clustering/prototype_labels.csv \
    --output results/breadth_steered.csv
