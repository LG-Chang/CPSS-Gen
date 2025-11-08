#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch caption images with Chat-UniVi (or compatible) and save to CSV.
This script mirrors the structure shown in your PDF, with completed lines and
garbled symbols fixed. Adjust model_path according to your local weights.
"""
import os
import csv
import torch
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Chat-UniVi imports (install per project docs)
from ChatUniVi.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)


@torch.inference_mode()
def process_images(folder_path: str, output_csv: str, model_path: str = "Chat-UniVi/Chat-UniVi-7B"):
    """Process images in all (nested) subfolders and save captions to CSV."""
    # ---------------------------- Load model ----------------------------
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path, model_base=None, model_name=model_name
    )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # Vision tower
    vision_tower = model.get_vision_tower()
    if not getattr(vision_tower, "is_loaded", True):
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    device = next(model.parameters()).device

    # ---------------------------- Walk images ----------------------------
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    rows = []
    folder_path = os.path.abspath(folder_path)
    print(f"[INFO] Scanning images under: {folder_path}")

    for root, dirs, files in os.walk(folder_path):
        dirs.sort()
        files = sorted(files)
        rel = os.path.relpath(root, folder_path)
        for filename in tqdm(files, desc=f"Processing {rel}"):
            if not filename.lower().endswith(image_exts):
                continue
            image_path = os.path.join(root, filename)
            try:
                # Build prompt with image token(s)
                qs = "Describe the image."
                if mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                conv = conv_templates["simple"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                # Tokenize and prepare image
                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).to(device)

                pil = Image.open(image_path).convert("RGB")
                image_tensor = image_processor.preprocess(pil, return_tensors="pt")["pixel_values"][0].to(device)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer)

                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half() if model.dtype == torch.float16 else image_tensor.unsqueeze(0),
                    do_sample=True,
                    temperature=0.2,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=512,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

                input_token_len = input_ids.shape[1]
                out_text = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
                if out_text.endswith(stop_str):
                    out_text = out_text[:-len(stop_str)].strip()

                rows.append([image_path, out_text])
            except (UnidentifiedImageError, OSError) as e:
                print(f"[WARN] Skipping unreadable image: {image_path} ({e})")
            except RuntimeError as e:
                print(f"[ERROR] Runtime while processing {image_path}: {e}")
            except Exception as e:
                print(f"[ERROR] {image_path}: {e}")

    # ---------------------------- Save CSV ----------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df = pd.DataFrame(rows, columns=["Image Path", "Caption"])
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    print(f"[INFO] Results saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch caption images with Chat-UniVi and save to CSV.")
    parser.add_argument("folder", help="Root folder containing images (scans recursively).")
    parser.add_argument("-o", "--output", default="captions.csv", help="Output CSV path.")
    parser.add_argument("--model_path", default="Chat-UniVi/Chat-UniVi-7B", help="Path or HF id to the model.")
    args = parser.parse_args()
    process_images(args.folder, args.output, model_path=args.model_path)


# python /mnt/data/univi_caption_batch.py /path/to/images -o /path/to/captions.csv --model_path Chat-UniVi/Chat-UniVi-7B
