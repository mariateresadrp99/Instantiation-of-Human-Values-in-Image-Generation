# Instantiation-of-Human-Values-in-Text-to-Image-Generation
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
 
**Main dependencies:**
- Python 3.12.1
- PyTorch
- `diffusers`
- `transformers`
- `sentence-transformers`
- `scikit-learn`
**Hardware:** Experiments were run on an NVIDIA A100 GPU with 24 CPU cores and 64 GB RAM.
 
---
 
## Repository Structure
 
```
├── prompts.csv                         # Full 140-prompt table (Appendix B)
├── taxonomy_definitions.json           # Social-configuration axis category descriptions
├── configs/
│   ├── sd35.yaml                       # SD 3.5 Medium generation config
│   ├── flux2.yaml                      # FLUX.2 9B generation config
│   ├── qwen_image.yaml                 # Qwen-Image generation config
│   └── gemini.yaml                     # Gemini 2.5 Flash config
├── generate_images.py                  # Step 1 — image generation
├── extract_vqa_descriptions.py         # Step 2 — VQA-based structured description
├── embed_descriptions.py               # Step 3 — embedding concatenated descriptions
├── cluster_prototypes.py               # Step 4 — K-means clustering + silhouette K selection
├── label_prototypes.py                 # Step 5 — LLM labeling (fine-grained + situation)
├── compute_breadth.py                  # Step 6 — inverse Simpson index (Neff)
├── map_taxonomy.py                     # Step 7 — social-configuration taxonomy mapping
├── steer_images.py                     # Step 8 — padding-position steering
├── utils/
│   ├── eos_mask.py                     # EOS/padding mask construction (per-model)
│   └── inverse_simpson.py              # Neff metric
├── analysis/
│   ├── breadth_results.py              # Tables 4–5, Figure 2, Wilcoxon tests
│   ├── demographic_analysis.py         # Gender/age distributions (Figures 10–13)
│   ├── human_reference_comparison.py   # Human vs. model Neff comparison (Figure 14)
│   └── social_config_plots.py          # Social-configuration stacked bar charts (Figures 4, 16–19)
└── run_pipeline.sh                     # End-to-end pipeline with checkpointing
```
 
---
 
## Running the Pipeline
 
### Full pipeline
 
```bash
bash run_pipeline.sh
```
 
### Step by step
 
**1. Generate images**
```bash
python generate_images.py \
  --prompts prompts.csv \
  --models sd35 flux2 qwen_image gemini \
  --n_seeds 50 \
  --output_dir data/images/
```
 
**2. Extract VQA descriptions**
```bash
python extract_vqa_descriptions.py \
  --image_dir data/images/ \
  --output_dir data/vqa_descriptions/
```
 
**3. Embed descriptions**
```bash
python embed_descriptions.py \
  --descriptions_dir data/vqa_descriptions/ \
  --output_dir data/embeddings/
```
 
**4. Cluster into visual prototypes**
```bash
python cluster_prototypes.py \
  --embeddings_dir data/embeddings/ \
  --k_range 10 200 \
  --output_dir data/prototypes/
```
 
**5. Label prototypes**
```bash
python label_prototypes.py \
  --prototypes_dir data/prototypes/ \
  --output_dir data/labels/
```
 
**6. Compute instantiation breadth**
```bash
python compute_breadth.py \
  --prototypes_dir data/prototypes/ \
  --labels_dir data/labels/ \
  --output_dir results/breadth/
```
 
**7. Map to social-configuration taxonomy**
```bash
python map_taxonomy.py \
  --prototypes_dir data/prototypes/ \
  --taxonomy taxonomy_definitions.json \
  --output_dir results/taxonomy/
```
 
**8. Steer images (optional)**
```bash
python steer_images.py \
  --value_item success \
  --model flux2 \
  --lambda_strength 1.5 \
  --tau 0.6 \
  --output_dir results/steered/
```
 
---
 
