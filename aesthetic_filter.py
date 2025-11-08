#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aesthetic score filtering with LAION aesthetic predictor + OpenCLIP.
- Recursively scans a folder of images
- Computes CLIP image embeddings
- Predicts aesthetic score with a small linear head
- Saves results to CSV (Image Path, Aesthetic Score)
"""
import os
import sys
import csv
import torch
import torch.nn as nn
import pandas as pd
from os.path import expanduser
from urllib.request import urlretrieve
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import open_clip

# ---------------------------- Config helpers ----------------------------
_AESTHETIC_URLS = {
    "vit_l_14": "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth",
    "vit_b_32": "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_b_32_linear.pth",
}

_CLIP_BACKBONES = {
    "vit_l_14": ("ViT-L-14", "laion2b_s32b_b82k", 768),
    "vit_b_32": ("ViT-B-32", "laion2b_e16", 512),
}


def get_aesthetic_model(clip_model: str = "vit_l_14") -> nn.Module:
    """Load the linear aesthetic predictor for a given CLIP backbone."""
    if clip_model not in _AESTHETIC_URLS or clip_model not in _CLIP_BACKBONES:
        raise ValueError(f"Unsupported CLIP model type: {clip_model}")

    home = expanduser("~")
    cache_folder = os.path.join(home, ".cache", "emb_reader")
    os.makedirs(cache_folder, exist_ok=True)
    path_to_model = os.path.join(cache_folder, f"sa_0_4_{clip_model}_linear.pth")
    if not os.path.exists(path_to_model):
        url_model = _AESTHETIC_URLS[clip_model]
        print(f"[INFO] Downloading aesthetic head to: {path_to_model}")
        urlretrieve(url_model, path_to_model)

    feat_dim = _CLIP_BACKBONES[clip_model][2]
    model = nn.Linear(feat_dim, 1)

    state_dict = torch.load(path_to_model, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_aesthetic_scores(folder_path: str, output_csv: str, clip_model: str = "vit_l_14"):
    """Evaluate aesthetic scores for all images in subfolders of a given folder."""
    if clip_model not in _CLIP_BACKBONES:
        raise ValueError(f"Unsupported CLIP model type: {clip_model}")

    backbone, pretrained, _ = _CLIP_BACKBONES[clip_model]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # OpenCLIP model + preprocess
    print(f"[INFO] Loading OpenCLIP backbone: {backbone} ({pretrained}) on {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, device=device)
    model.to(device).eval()

    # Aesthetic head
    a_head = get_aesthetic_model(clip_model=clip_model).to(device).eval()

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    results = []

    # Walk through all subfolders (sorted for determinism)
    folder_path = os.path.abspath(folder_path)
    print(f"[INFO] Scanning images under: {folder_path}")
    for root, dirs, files in os.walk(folder_path):
        dirs.sort()
        files = sorted(files)
        for filename in tqdm(files, desc=f"Processing {os.path.relpath(root, folder_path)}"):
            if filename.lower().endswith(image_exts):
                image_path = os.path.join(root, filename)
                try:
                    img = Image.open(image_path).convert("RGB")
                    image_tensor = preprocess(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_features = model.encode_image(image_tensor)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        score = a_head(image_features).item()

                    results.append([image_path, float(score)])
                except (UnidentifiedImageError, OSError) as e:
                    print(f"[WARN] Skipping unreadable image: {image_path} ({e})")
                except Exception as e:
                    print(f"[ERROR] {image_path}: {e}")

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df = pd.DataFrame(results, columns=["Image Path", "Aesthetic Score"])
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    print(f"[INFO] Results saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute aesthetic scores for images using OpenCLIP.")
    parser.add_argument("folder", help="Root folder containing images (scans recursively).")
    parser.add_argument("-o", "--output", default="aesthetic_scores.csv", help="Output CSV path.")
    parser.add_argument("--clip", default="vit_l_14", choices=list(_CLIP_BACKBONES.keys()), help="CLIP backbone.")
    args = parser.parse_args()
    evaluate_aesthetic_scores(args.folder, args.output, clip_model=args.clip)

# python /mnt/data/aesthetic_filter.py /path/to/images -o /path/to/aesthetic_scores.csv --clip vit_l_14
