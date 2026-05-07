#!/usr/bin/env bash
# run_pipeline.sh
# ----------------
# End-to-end pipeline: generation → VQA extraction → clustering → (optional) steering.
# Each step is checkpointed — re-running skips already completed work.
#
# Prerequisites:
#   pip install -r requirements.txt
#   export REPLICATE_API_TOKEN=<your_token>   # only needed for Gemini
#
# Usage:
#   bash run_pipeline.sh

set -euo pipefail

CSV="prompts.csv"
IMAGES_PER_PROMPT=50
N_SEEDS=50

# ── Step 1: Image generation ───────────────────────────────────────────────────
echo "=== [1/4] Generating images ==="

python generation/generate_images_flux2.py \
    --csv "$CSV" --out data/images/flux2 \
    --images-per-prompt "$IMAGES_PER_PROMPT" --which both

python generation/generate_images_qwen.py \
    --csv "$CSV" --out data/images/qwen \
    --images-per-prompt "$IMAGES_PER_PROMPT" --aspect-ratio 1:1 --which both

python generation/generate_images_sd35.py \
    --csv "$CSV" --out data/images/sd35 \
    --images-per-prompt "$IMAGES_PER_PROMPT" --which both

# Gemini requires REPLICATE_API_TOKEN
python generation/generate_images_gemini.py \
    --csv "$CSV" --out data/images/gemini \
    --images-per-prompt "$IMAGES_PER_PROMPT" --which both

# ── Step 2: VQA extraction ─────────────────────────────────────────────────────
echo "=== [2/4] Extracting VQA descriptions ==="

for MODEL in flux2 qwen sd35 gemini; do
    for POLARITY in positive_prompt negative_prompt; do
        OUT="data/vqa/results_${MODEL}_${POLARITY}.json"
        if [ -f "$OUT" ]; then
            echo "  [SKIP] $OUT already exists."
        else
            python vqa/extract_vqa_descriptions.py \
                --csv "$CSV" \
                --img_root "data/images/$MODEL" \
                --output "$OUT" \
                --prompt_type "$POLARITY"
        fi
    done
done

# ── Step 3: Clustering ─────────────────────────────────────────────────────────
echo "=== [3/4] Clustering visual prototypes ==="

python clustering/cluster_prototypes.py \
    --input_positive \
        data/vqa/results_gemini_positive_prompt.json \
        data/vqa/results_flux2_positive_prompt.json \
        data/vqa/results_qwen_positive_prompt.json \
        data/vqa/results_sd35_positive_prompt.json \
    --input_negative \
        data/vqa/results_gemini_negative_prompt.json \
        data/vqa/results_flux2_negative_prompt.json \
        data/vqa/results_qwen_negative_prompt.json \
        data/vqa/results_sd35_negative_prompt.json \
    --outdir results/clustering \
    --sample_per_group "$N_SEEDS" \
    --k_min 3 --k_max 200 \
    --sample_col prompt_text

# ── Step 4: Steering (optional — run manually per value) ───────────────────────
echo "=== [4/4] Steering (example — edit archetypes as needed) ==="
echo "  Skipping automatic steering — run manually, e.g.:"
echo ""
echo "  python steering/steer_flux2.py \\"
echo "    --prompt 'a successful person' \\"
echo "    --archetypes \\"
echo "      'a group of people of different ages celebrating at home' \\"
echo "      'a person in workwear at a construction site' \\"
echo "    --out results/steered/flux2/successful \\"
echo "    --seeds \$(seq 42 81 | tr '\\n' ' ')"
echo ""

echo "=== Pipeline complete ==="
