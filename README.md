# Instantiation-of-Human-Values-in-Image-Generation
![Evaluation Methodology](./overview_framework.png)
Text-to-Image (T2I) models are increasingly embedded in everyday visual culture, yet little is known about how they translate abstract human values into visual scenes. In this paper, we introduce a framework, grounded in Schwartz’s theory of basic human values, for studying \textit{value instantiation}: the process by which ideals such as \textit{success}, \textit{care}, or \textit{freedom} are visually realized. Using person-centered prompts from 70 positive and negative value items, we generate 28,000 images across four state-of-the-art T2I models. We identify visual prototypes, recurring combinations of subjects, appearances, actions, and settings, and use them to quantify how narrowly models instantiate values. We then map visual prototypes onto a taxonomy of social configurations spanning \textit{setting}, \textit{sociality}, \textit{mode of engagement}, and \textit{social register}, revealing what is systematically foregrounded or omitted. Our results show that T2I models rely on a narrow set of recurring value instantiations. They default to an individualistic, adult-centric worldview, reproducing value-specific associations, such as achievement tied to formal institutional settings or hedonism shown as luxury consumption, while marginalizing everyday labor and non-adult perspectives. Finally, we demonstrate that these structural gaps can be translated into steering signals that broaden outputs without modifying the original prompt.

## Setup
 
### Requirements
 
```bash
pip install -r requirements.txt
```
 
Or with conda:
 
```bash
conda env create -f environment.yml
conda activate value-instantiation
```
 
**Main dependencies:** Python 3.12.1, PyTorch, `diffusers`, `transformers`, `sentence-transformers`, `scikit-learn`
 
**Hardware:** Experiments were run on an NVIDIA A100 GPU with 24 CPU cores and 64 GB RAM.
 
---
 
## Repository Structure
 
```
├── prompts.csv                              # Full 140-prompt table (Appendix B)
├── taxonomy_definitions.json               # Social-configuration axis category descriptions
├── requirements.txt
├── meth_framework.png
│
├── image_generation/
│   ├── generate_images_flux2.py             # FLUX.2 Klein 9B
│   ├── generate_images_gemini.py            # Gemini 2.5 Flash (via Replicate)
│   ├── generate_images_qwen.py              # Qwen-Image
│   └── generate_images_sd35.py             # Stable Diffusion 3.5 Medium
│
├── extract_vqa_descriptions.py             # VQA-based structured image description
│
├── clustering/
│   └── cluster_prototypes.py               # K-means clustering + silhouette K selection
│
├── steering/
│   ├── steer_flux2.py                       # Phased EOS-injection steering for FLUX.2
│   └── steer_sd35.py                        # EOS-injection steering for SD3.5
│
└── analysis/
    └── map_archetypes_to_human.py           # Cosine mapping of model archetypes to human clusters
```
 
---
 
## Running the Pipeline
 
### Step 1 — Generate images
 
Each script is **resumable**: it skips already-generated images and tops up to the target count.
 
**FLUX.2 Klein**
```bash
python image_generation/generate_images_flux2.py \
  --csv prompts.csv \
  --out data/images/flux2 \
  --images-per-prompt 50 \
  --which both
```
 
**Qwen-Image**
```bash
python image_generation/generate_images_qwen.py \
  --csv prompts.csv \
  --out data/images/qwen \
  --images-per-prompt 50 \
  --aspect-ratio 1:1 \
  --which both
```
 
**Stable Diffusion 3.5 Medium**
```bash
python image_generation/generate_images_sd35.py \
  --csv prompts.csv \
  --out data/images/sd35 \
  --images-per-prompt 50 \
  --which both
```
 
**Gemini 2.5 Flash** *(requires `REPLICATE_API_TOKEN`)*
```bash
export REPLICATE_API_TOKEN=<your_token>
 
python image_generation/generate_images_gemini.py \
  --csv prompts.csv \
  --out data/images/gemini \
  --images-per-prompt 50 \
  --which both
```
 
All four generators share the same output structure:
```
data/images/<model>/<schwartz_domain>/<value>/positive_prompt/*.png
data/images/<model>/<schwartz_domain>/<value>/negative_prompt/*.png
```
 
To parallelize across GPUs, use `--start-idx` and `--end-idx` to slice the CSV by row range.
 
---
 
### Step 2 — Extract VQA descriptions
 
Extracts structured descriptors (demographics, appearance, objects, setting, action) from each generated image using Qwen2-VL-7B-Instruct.
 
```bash
python extract_vqa_descriptions.py \
  --csv prompts.csv \
  --img_root data/images/flux2 \
  --output data/vqa/results_flux2_positive.json \
  --prompt_type positive_prompt
```
 
Run for each model and polarity:
```bash
for MODEL in flux2 qwen sd35 gemini; do
  for POLARITY in positive_prompt negative_prompt; do
    python extract_vqa_descriptions.py \
      --csv prompts.csv \
      --img_root data/images/$MODEL \
      --output data/vqa/results_${MODEL}_${POLARITY}.json \
      --prompt_type $POLARITY
  done
done
```
 
---
 
### Step 3 — Cluster visual prototypes
 
Pools all VQA descriptions and clusters them jointly into global visual prototypes using K-means with silhouette-based K selection (K ∈ [3, 200]).
 
```bash
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
  --sample_per_group 50 \
  --k_min 3 --k_max 200 \
  --sample_col prompt_text
```
 
Outputs per polarity:
```
results/clustering/
  summaries/summary__GLOBAL__<scope>.txt
  positive/
    domain/      domain_matrix__<scope>.csv
    value/       value_matrix__<scope>.csv
    prompt_text/<value>/keyword_matrix__<value>__<scope>.csv
  negative/
    (identical structure)
```
 
---
 
### Step 4 — Steer images
 
Steers generation toward underrepresented social configurations using EOS-position embedding injection (Kim et al., CVPR 2025), without modifying the original prompt.
 
**FLUX.2 Klein**
```bash
python steering/steer_flux2.py \
  --prompt "a successful person" \
  --archetypes \
      "a group of people of different ages celebrating at home" \
      "a person in workwear at a construction site" \
      "friends sitting together at a dinner table" \
  --scale 1.5 \
  --tau 0.4 \
  --out results/steered/flux2/successful \
  --seeds $(seq 42 81 | tr '\n' ' ')
```
 
**Stable Diffusion 3.5**
```bash
python steering/steer_sd35.py \
  --prompt "a successful person" \
  --archetypes \
      "a group of people of different ages celebrating at home" \
      "a person in workwear at a construction site" \
  --scale 1.0 \
  --tau 0.4 \
  --out results/steered/sd35/successful \
  --seeds $(seq 42 81 | tr '\n' ' ')
```
 
Default hyperparameters per model:
 
| Model  | `--scale` (λ) | `--tau` (τ) | `--guidance` |
|--------|:---:|:---:|:---:|
| FLUX.2 | 1.5 | 0.6 | 4.0 |
| SD3.5  | 1.0 | 0.4 | 7.0 |
| Qwen-Image | 1.5 | 0.4 | 6.0 |
 
---
 
 
