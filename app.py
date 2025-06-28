#!/usr/bin/env python3
"""
RESTful API for Fast Real-Time Image Search

This FastAPI application provides endpoints for:
  - Uploading images
  - Zero-shot segmentation via CLIPSeg
  - Generating CLIP embeddings for segmented images
  - Real-time similarity search via FAISS, returning matching images

OpenAPI docs available at:
  - JSON: /openapi.json
  - Swagger UI: /docs
  - ReDoc UI: /redoc

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import base64
import uuid
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import (
    FastAPI, UploadFile, File, HTTPException,
    Query, status
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
from transformers import (
    CLIPSegProcessor, CLIPSegForImageSegmentation,
    CLIPProcessor, CLIPModel
)

# --- Configuration ---
BASE_DIR   = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MASK_DIR   = BASE_DIR / "masks"
EMB_DIR    = BASE_DIR / "data/train/embeddings"
INDEX_PATH = BASE_DIR / "data/train/index/faiss_index"
MAP_PATH   = BASE_DIR / "data/train/mapping/id_mapping.npy"

for d in (UPLOAD_DIR, MASK_DIR, EMB_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load models ---
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model     = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE)
clipseg_model.eval()

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_model.eval()

# --- Load FAISS index ---
if INDEX_PATH.exists() and MAP_PATH.exists():
    faiss_index = faiss.read_index(str(INDEX_PATH))
    id_mapping  = np.load(str(MAP_PATH))
else:
    faiss_index = None
    id_mapping  = None

# --- API metadata and tags ---
tags_metadata = [
    {"name": "upload",  "description": "Upload an image for later processing."},
    {"name": "segment", "description": "Perform zero-shot segmentation."},
    {"name": "embed",   "description": "Generate CLIP embeddings for segmented images."},
    {"name": "search",  "description": "Search similar images by embedding."},
    {"name": "index",   "description": "Build or rebuild the FAISS index."},
]

app = FastAPI(
    title="Big Cat Image Search API",
    version="1.0",
    description="API for uploading, segmenting, embedding, and searchable image engine.",
    openapi_tags=tags_metadata
)

# --- Response Models ---
class UploadResponse(BaseModel):
    id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174000")
    filename: str = Field(..., example="123e4567-e89b-12d3-a456-426614174000.jpg")

class EmbedResponse(BaseModel):
    id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174000")
    embedding_path: str = Field(..., example="/data/train/embeddings/123... .npy")

class SearchResult(BaseModel):
    id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174001")
    score: float = Field(..., example=0.95)
    image: str = Field(..., description="Base64-encoded image data URI")

# --- Endpoints ---
@app.post(
    "/upload-image",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["upload"],
    summary="Upload an image for processing"
)
async def upload_image(
    file: UploadFile = File(..., description="Image file (.jpg, .jpeg, .png)")
):
    """
    Uploads an image and returns a unique image ID for future calls.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type"
        )
    image_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{image_id}{ext}"
    with path.open("wb") as f:
        f.write(await file.read())
    return {"id": image_id, "filename": path.name}

@app.post(
    "/segment",
    response_class=FileResponse,
    tags=["segment"],
    summary="Zero-shot segmentation via CLIPSeg"
)
async def segment_image(
    id: str = Query(..., description="Image ID returned from /upload-image"),
    prompt: str = Query(..., description="Text prompt, e.g. 'a tiger'")
):
    """
    Runs CLIPSeg zero-shot segmentation on the uploaded image using the given prompt.

    Returns a PNG mask file.
    """
    matches = list(UPLOAD_DIR.glob(f"{id}.*"))
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image ID not found"
        )
    image = Image.open(matches[0]).convert("RGB")
    inputs = clipseg_processor(
        text=[prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clipseg_model(**inputs)
    mask_logits = outputs.logits[0]
    mask_prob   = torch.sigmoid(mask_logits).unsqueeze(0).unsqueeze(0)
    w, h = image.size
    mask_resized = F.interpolate(
        mask_prob, size=(h, w), mode="bilinear", align_corners=False
    )[0,0]
    mask_np = (mask_resized.cpu().numpy() * 255).astype(np.uint8)
    out_path = MASK_DIR / f"{id}_mask.png"
    Image.fromarray(mask_np).save(out_path)
    return str(out_path)

@app.post(
    "/embed",
    response_model=EmbedResponse,
    tags=["embed"],
    summary="Generate CLIP embedding for a segmented image"
)
async def embed_image(
    id: str = Query(..., description="Image ID from /upload-image and /segment")
):
    """
    Generates and saves a normalized CLIP embedding for the segmented image.

    Returns the path to the .npy embedding file.
    """
    matches = list(UPLOAD_DIR.glob(f"{id}.*"))
    mask_fp = MASK_DIR / f"{id}_mask.png"
    if not matches or not mask_fp.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image or mask not found"
        )
    image = Image.open(matches[0]).convert("RGB")
    mask  = Image.open(mask_fp).convert("L")
    arr   = np.array(image, dtype=np.float32)
    m_arr = np.array(mask, dtype=np.float32) / 255.0
    arr  *= m_arr[..., None]
    masked = Image.fromarray(arr.astype(np.uint8))
    inputs = clip_processor(images=masked, return_tensors="pt")
    vals   = inputs.pixel_values.to(DEVICE)
    with torch.no_grad():
        feat = clip_model.get_image_features(vals)
    feat_norm = feat / feat.norm(p=2, dim=-1, keepdim=True)
    emb = feat_norm.cpu().numpy().squeeze(0)
    emb_fp = EMB_DIR / f"{id}.npy"
    np.save(str(emb_fp), emb)
    return {"id": id, "embedding_path": str(emb_fp)}

@app.get(
    "/search",
    response_model=list[SearchResult],
    tags=["search"],
    summary="Search for similar images"
)
async def search_image(
    id: str = Query(..., description="Image ID to query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of top matches to return")
):
    """
    Queries the FAISS index for the top-K similar images to the given embedding.

    Returns a list of IDs, similarity scores, and Base64-encoded images.
    """
    if faiss_index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index not built"
        )
    emb_fp = EMB_DIR / f"{id}.npy"
    if not emb_fp.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Embedding not found; call /embed first"
        )
    qvec = np.load(str(emb_fp)).astype(np.float32)[None, :]
    D, I = faiss_index.search(qvec, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        match_id = id_mapping[idx]
        found = list(UPLOAD_DIR.glob(f"{match_id}.*"))
        if found:
            data = base64.b64encode(found[0].read_bytes()).decode()
            ext = found[0].suffix.lstrip('.')
            uri = f"data:image/{ext};base64,{data}"
        else:
            uri = None
        results.append({"id": match_id, "score": float(score), "image": uri})
    return results

@app.post(
    "/build-index",
    response_model=dict,
    tags=["index"],
    summary="Build FAISS index from existing embeddings"
)
async def build_index_endpoint():
    """
    Reads all .npy embeddings, builds a FAISS IVF index, and saves it.

    Returns the number of indexed embeddings.
    """
    files = list(EMB_DIR.glob("*.npy"))
    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No embeddings found"
        )
    embs, keys = [], []
    for f in files:
        keys.append(f.stem)
        embs.append(np.load(str(f)))
    mat = np.stack(embs).astype(np.float32)
    faiss.normalize_L2(mat)
    dim = mat.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)
    idx.train(mat)
    idx.add(mat)
    faiss.write_index(idx, str(INDEX_PATH))
    np.save(str(MAP_PATH), np.array(keys))
    global faiss_index, id_mapping
    faiss_index = idx
    id_mapping  = np.array(keys)
    return {"indexed": len(keys)}
