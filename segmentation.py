from pathlib import Path
from PIL import Image
import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Load CLIPSeg model and processor
threshold = 0.08
alpha = 0.5
device = torch.device('cpu')
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.to(device)
model.eval()


# Mapping for class prompts
prompt_map = {
    "tigre": "a tigre",
    "leopard": "a leopard",
    "guepard": "a guepard",
    "jaguar": "the animal"  # updated it since jaguar is not getting correct masks
    }


# Define the dataset root directory (change this to your actual dataset path)
dataset_root = Path("data/train/image/")

# Define output directory for segmentation results
output_root = Path("data/train/output_masks")
output_root.mkdir(parents=True, exist_ok=True)  # create it if it doesn't exist

# Get class subdirectories (each directory in dataset_root is a class)
class_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]

# Iterate over each class folder and each image file
for class_dir in class_dirs:
    class_name = class_dir.name
    print(f"Processing class: {class_name}")
    # Create corresponding class folder in output directory
    output_class_dir = output_root / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = prompt_map.get(class_name, f"a {class_name}")
    print(f"Processing class '{class_name}' with prompt '{prompt_text}'...")
    # Loop through image files in this class directory
    for img_path in class_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in ".jpg":
            # Open the image using PIL
            image = Image.open(img_path).convert("RGB")
            # Prepare inputs
            inputs = processor(
                text=[prompt_text],
                images=[image],
                padding=True,           # enable dynamic padding
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0]  # shape (H, W)
            probs = torch.sigmoid(logits)

            # Upsample to original image size
            w, h = image.size
            probs_upsampled = F.interpolate(
                probs.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False
            ).squeeze()
            mask = probs_upsampled.cpu().numpy()
            binary_mask = mask > threshold

            # Create colored overlay
            img_arr = np.array(image, dtype=np.uint8)
            overlay = img_arr.copy().astype(np.float32)
            color = np.array([255, 0, 0], dtype=np.float32)
            overlay[binary_mask] = overlay[binary_mask] * (1 - alpha) + color * alpha
            overlay = overlay.astype(np.uint8)
            overlay_img = Image.fromarray(overlay)

            # Save overlay
            overlay_path = output_class_dir / f"{img_path.stem}_overlay.png"
            overlay_img.save(overlay_path)

            # Save raw mask (optional)
            mask_img = Image.fromarray((binary_mask.astype(np.uint8) * 255))
            mask_path = output_class_dir / f"{img_path.stem}_mask.png"
            mask_img.save(mask_path)

            print(f"Saved: {overlay_path.name}, {mask_path.name}")
