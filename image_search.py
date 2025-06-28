#!/usr/bin/env python3
"""
Fast Real-Time Image Retrieval System using CLIP embeddings and FAISS.


Configuration:
    # Mode: "build_index" or "query"
    MODE          = "build_index"

    # Paths for build_index mode
    EMBEDDINGS_PATH = Path("/absolute/path/to/embeddings.npz")
    INDEX_PATH      = Path("/absolute/path/to/index.faiss")
    MAPPING_PATH    = Path("/absolute/path/to/id_mapping.npy")

    # Paths and parameters for query mode
    QUERY_IMAGE  = Path("/absolute/path/to/query_image.jpg")
    TOP_K         = 5

    # Index settings (used in build_index)
    NORMALIZE     = True          # L2-normalize for cosine similarity
    INDEX_TYPE    = "ivf_flat"   # "flat" or "ivf_flat"
    NLIST         = 100           # number of clusters if using ivf_flat

    # CLIP model for query embedding
    MODEL_NAME    = "openai/clip-vit-base-patch32"
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

Modes:
    - build_index : Load embeddings and build/save FAISS index and mapping.
    - query       : Embed QUERY_IMAGE and retrieve TOP_K nearest neighbors.
"""
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path

# Configuration (hardcoded)
MODE =  "query"  # or "build_index" or "query"

EMBEDDINGS_PATH = Path("data/train/embeddings/masked_clip_embeddings.npz")
INDEX_PATH      = Path("data/train/index/faiss.index")
MAPPING_PATH    = Path("data/train/mapping/id_mapping.npy")

QUERY_IMAGE = Path("../data/test/jaguar/imagem019_jpg.rf.8ab29dfb2044dc811581ccb0e613aeae.jpg")
TOP_K       = 1

NORMALIZE  = True
INDEX_TYPE = "ivf_flat"
NLIST      = 100

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_index():
    # Load embeddings
    data = np.load(EMBEDDINGS_PATH)
    keys = list(data.keys())
    embeddings = np.stack([data[k] for k in keys]).astype(np.float32)
    dim = embeddings.shape[1]

    # Normalize if required
    if NORMALIZE:
        faiss.normalize_L2(embeddings)

    # Choose FAISS index
    if INDEX_TYPE == "flat":
        index = faiss.IndexFlatIP(dim) if NORMALIZE else faiss.IndexFlatL2(dim)
    else:
        quantizer = faiss.IndexFlatIP(dim) if NORMALIZE else faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, NLIST,
                                   faiss.METRIC_INNER_PRODUCT if NORMALIZE else faiss.METRIC_L2)
        index.train(embeddings)
    index.add(embeddings)

    # Save index and mapping
    # Save index
    faiss.write_index(index, str(INDEX_PATH))

    # Save mapping
    if MAPPING_PATH.is_dir():
        np.save(str(MAPPING_PATH / 'id_mapping.npy'), np.array(keys))
    else:
        np.save(str(MAPPING_PATH), np.array(keys))

    print(f"Built FAISS index with {len(keys)} vectors. Index file: {INDEX_PATH}")


def embed_query_image():
    # Load CLIP model and processor
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME,revision="d15b5f29721ca72dac15f8526b284be910de18be")
    model.to(DEVICE)
    model.eval()

    # Load and preprocess query image
    image = Image.open(QUERY_IMAGE).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(DEVICE)

    with torch.no_grad():
        feat = model.get_image_features(pixel_values)
    qvec = feat.cpu().numpy().astype(np.float32)

    if NORMALIZE:
        faiss.normalize_L2(qvec)
    return qvec


def query_index():
    # Load FAISS index and mapping
    index = faiss.read_index(str(INDEX_PATH))
    keys = np.load(str(MAPPING_PATH))

    # Embed the query image
    qvec = embed_query_image()

    # Search
    distances, indices = index.search(qvec, TOP_K)
    for rank, idx in enumerate(indices[0], start=1):
        print(f"Rank {rank}: {keys[idx]} (score={distances[0][rank-1]:.4f})")


def main():
    if MODE == "build_index":
        build_index()
    elif MODE == "query":
        query_index()
    else:
        raise ValueError(f"Unknown MODE '{MODE}'")


if __name__ == "__main__":
    main()
