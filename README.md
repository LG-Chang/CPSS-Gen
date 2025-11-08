# CPSS-Gen: Synthetic Chemical Plant Safety Suite

This repository packages the sample image set and tooling used to synthesize, caption, and filter surveillance imagery for chemical-plant safety monitoring. Use it to explore the data categories and to run the three provided utilities for prompt generation, Chat-UniVi captioning, and LAION aesthetic scoring.

## Dataset

The sample set covers eight behavior / hazard categories relevant to chemical plant safety monitoring. Each cell in the gallery links to one illustrative image:

<table>
  <tr>
    <td align="center">Call<br><img src="img&#47;call.jpg" alt="call" width="200"><br><sub>Handheld device use on duty</sub></td>
    <td align="center">Flame<br><img src="img&#47;flame.jpg" alt="flame" width="200"><br><sub>Open flame near equipment</sub></td>
    <td align="center">Oil Leakage<br><img src="img&#47;oil_leakage.jpg" alt="oil leakage" width="200"><br><sub>Liquid spill / leakage</sub></td>
    <td align="center">Safety Belt<br><img src="img&#47;safety_belt.jpg" alt="safety belt" width="200"><br><sub>Fall-protection usage</sub></td>
  </tr>
  <tr>
    <td align="center">Safety Cap<br><img src="img&#47;safety_cap.jpg" alt="safety cap" width="200"><br><sub>Helmet compliance</sub></td>
    <td align="center">Sleeping<br><img src="img&#47;sleeping.jpg" alt="sleeping" width="200"><br><sub>Fatigue / inattentive state</sub></td>
    <td align="center">Smoking<br><img src="img&#47;smoking.jpg" alt="smoking" width="200"><br><sub>Smoking in restricted area</sub></td>
    <td align="center">Trip<br><img src="img&#47;trip.jpg" alt="trip" width="200"><br><sub>Tripping or slipping event</sub></td>
  </tr>
</table>

A lightweight sample pack of per-class images and labels is included for quick prototyping and validation.

## Code

### `prompt_generation.py`

Implements a six-layer prompt template (camera configuration, worker behavior, environment + hazard context, lighting, scale, and occlusion) to produce balanced textual descriptions for diffusion / ControlNet data generation.

### `univi_caption_batch.py`

Provides batch captioning via Chat-UniVi, recursively scanning folders, assembling multimodal prompts, and exporting a CSV with image paths and generated descriptions.

### `aesthetic_filter.py`

Computes OpenCLIP embeddings and applies LAION's linear aesthetic head to score images, enabling automatic filtering of low-quality generations prior to detector training.

## How To Use?

1. Install the core Python dependencies (add any CUDA-specific wheels as needed):
   ```bash
   pip install torch pandas pillow tqdm open_clip_torch
   ```
   For captioning, also install [Chat-UniVi](https://github.com/PKU-YuanGroup/Chat-UniVi) following its repository instructions.
2. Generate balanced text prompts with `prompt_generation.py`:
   ```bash
   python prompt_generation.py
   ```
   (Outputs 500 prompts per unsafe-behavior class to `output/balanced_behavior_500.txt` by default.)
3. Caption your image folders with `univi_caption_batch.py`, pointing `--model_path` to your Chat-UniVi checkpoint or Hub ID:
   ```bash
   python univi_caption_batch.py /path/to/images \
       -o captions.csv \
       --model_path Chat-UniVi/Chat-UniVi-7B
   ```
4. Filter or rank the resulting imagery with `aesthetic_filter.py` to keep the most visually pleasing samples for downstream detectors:
   ```bash
   python aesthetic_filter.py /path/to/images \
       -o aesthetic_scores.csv \
       --clip vit_l_14
   ```
