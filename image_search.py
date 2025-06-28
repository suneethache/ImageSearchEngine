#!/usr/bin/env python3
"""
Fast Real-Time Image Retrieval System using CLIP embeddings and FAISS.
"""
import argparse
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path

# Default configuration
DEFAULT_EMBEDDINGS = Path("data/train/embeddings/masked_clip_embeddings.npz")
DEFAULT_INDEX      = Path("data/train/index/faiss.index")
DEFAULT_MAPPING    = Path("data/train/mapping/id_mapping.npy")
DEFAULT_QUERY_IMG  = Path("../data/test/jaguar/imagem019_jpg.rf.8ab29dfb2044dc811581ccb0e613aeae.jpg")
DEFAULT_TOP_K      = 1

NORMALIZE  = True
INDEX_TYPE = "ivf_flat"  # or "flat"
NLIST      = 100
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_index(emb_path: Path, idx_path: Path, map_path: Path):
    # Load embeddings
    data = np.load(emb_path)
    keys = list(data.keys())
    embeddings = np.stack([data[k] for k in keys]).astype(np.float32)
    dim = embeddings.shape[1]

    # Normalize if required
    if NORMALIZE:
        faiss.normalize_L2(embeddings)

    # Create FAISS index
    if INDEX_TYPE == "flat":
        index = faiss.IndexFlatIP(dim) if NORMALIZE else faiss.IndexFlatL2(dim)
    else:
        quantizer = faiss.IndexFlatIP(dim) if NORMALIZE else faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(
            quantizer, dim, NLIST,
            faiss.METRIC_INNER_PRODUCT if NORMALIZE else faiss.METRIC_L2
        )
        index.train(embeddings)

    # Add embeddings and save
    index.add(embeddings)
    faiss.write_index(index, str(idx_path))

    # Save mapping
    np.save(str(map_path), np.array(keys, dtype=object))
    print(f"Built FAISS index with {len(keys)} vectors → {idx_path}")


def embed_query_image(query_img: Path):
    # Load CLIP
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE).eval()

    # Preprocess image
    image = Image.open(query_img).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(DEVICE)

    # Extract features
    with torch.no_grad():
        feats = model.get_image_features(pixel_values)
    qvec = feats.cpu().numpy().astype(np.float32)

    if NORMALIZE:
        faiss.normalize_L2(qvec)
    return qvec


def query_index(idx_path: Path, map_path: Path, query_img: Path, top_k: int):
    # Load index and mapping
    index = faiss.read_index(str(idx_path))
    keys = np.load(str(map_path), allow_pickle=True)

    # Embed and search
    qvec = embed_query_image(query_img)
    distances, indices = index.search(qvec, top_k)

    # Display results
    for rank, idx in enumerate(indices[0], start=1):
        score = distances[0][rank-1]
        print(f"Rank {rank}: {keys[idx]} (score={score:.4f})")


def parse_args():
    p = argparse.ArgumentParser(
        description="Build or query a FAISS index over CLIP embeddings"
    )
    p.add_argument(
        "-m", "--mode",
        choices=["build_index", "query"],
        required=True,
        help="Operation mode: build_index or query"
    )
    p.add_argument(
        "--embeddings-path",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help="Path to .npz of CLIP embeddings"
    )
    p.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX,
        help="Path to read/write the FAISS index"
    )
    p.add_argument(
        "--mapping-path",
        type=Path,
        default=DEFAULT_MAPPING,
        help="Path to read/write the ID mapping (.npy)"
    )
    p.add_argument(
        "-q", "--query-image",
        type=Path,
        default=DEFAULT_QUERY_IMG,
        help="Path to query image (only used in query mode)"
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of nearest neighbors to return (query mode)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "build_index":
        build_index(
            emb_path=args.embeddings_path,
            idx_path=args.index_path,
            map_path=args.mapping_path
        )
    else:  # query
        query_index(
            idx_path=args.index_path,
            map_path=args.mapping_path,
            query_img=args.query_image,
            top_k=args.top_k
        )


if __name__ == "__main__":
    main()
