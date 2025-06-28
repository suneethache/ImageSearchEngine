""" Generated the embeddings of segmented masks and saves the embeddings"""
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"

dataset_root = Path('data/train/images/')
masks_root   = Path('data/train/output_masks/')
output_dir   = Path('data/train/embeddings/')
output_dir.mkdir(parents=True, exist_ok=True)

processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name, revision="d15b5f29721ca72dac15f8526b284be910de18be")
model.to(device)
model.eval()

embeddings = {}
valid_exts = {'.jpg', '.jpeg', '.png'}
threshold = 0.08

for class_dir in dataset_root.iterdir():
    if not class_dir.is_dir():
        continue
    for img_path in class_dir.iterdir():
        if img_path.suffix.lower() not in valid_exts:
            continue

        mask_path = masks_root / class_dir.name / f"{img_path.stem}_mask.png"
        if not mask_path.exists():
            print(f"Warning: mask not found for {img_path.name}, skipping.")
            continue

        # Load & apply mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        img_arr  = np.array(img,  dtype=np.float32)
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_bin = (mask_arr > threshold).astype(np.float32)[..., None]
        masked   = (img_arr * mask_bin).astype(np.uint8)
        masked_img = Image.fromarray(masked)

        # Embeddings
        inputs = processor(images=masked_img, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

        embeddings[img_path.stem] = feats.cpu().numpy().squeeze(0)
        print(f"→ {img_path.name}")

# Saving the embeddings
out_file = output_dir / "masked_clip_embeddings.npz"
np.savez_compressed(out_file, **embeddings)
print(f"Saved {len(embeddings)} embeddings to {out_file}")
