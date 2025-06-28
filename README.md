# Real time Image search engine using CLIP models

A fast, real-time image search engine built with segmentation, CLIP embeddings, FAISS indexing, and a RESTful API all containerized with Docker.  

## Project Overview

This project implements an end-to-end image search pipeline:

1. **Zero-shot Segmentation**  
   Segment image regions via CLIPSeg to isolate objects of interest.  
2. **Few-Shot Embeddings**  
   Generate and fine-tune CLIP embeddings on the segmented masks
3. **FAISS Indexing & Search**  
   Build a FAISS IVF index for efficient nearest-neighbor search over the embedding space
4. **REST API**  
   Quickly upload images, segment, embed, and query via HTTP endpoints powered by FastAPI 
5. **Dockerization**  
   Containerize the entire stack and orchestrate with Docker Compose for one-command deployment

---
## 🧪 Dataset

The dataset that has been used for this project can be found : https://universe.roboflow.com/azza-te8hj/my-first-project-klagq
Download and extract images into data/train/image/ before running segmentation

---
## 📁 Project Structure
```
├── app.py                     # FastAPI application
├── segmentation.py            # Batch segmentation script
├── generate_embeddings.py     # Batch CLIP embedding script
├── image_search.py            # CLI for FAISS index & query
├── Dockerfile                 # API container build
├── docker-compose.yml         # Compose orchestration
├── data/
│   ├── train/
│   │   ├── image/             # Raw dataset images
│   │   ├── output_masks/      # Segmentation outputs
│   │   ├── embeddings/        # .npy & .npz embeddings
│   │   ├── index/             # FAISS index files
│   │   └── mapping/           # ID mapping arrays
├── uploads/                   # Runtime uploads (mounted)
├── masks/                     # Runtime masks (mounted)
├── requirements.txt           # Contains all the necessary library installtions
└── README.md
```

Note that I have kept the train data corresponding masks, embeddings, index, and mapping files in seperate folders under train dataset and all the scripts.
All the scripts that need embeddings or any other file will find in train folder, so pleasedo the same or update the paths in image_seach.py file

---
## Running the project

## 1. Installing requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

By running the above lines of code, everything that needed are installed.

## 2. To Create Batch Segmentation
```bash
python segmentation.py
```
Processes images under data/train/image/, applies CLIPSeg, and writes overlays & masks to data/train/output_masks/

## 3. To Generate embeddings of segmented masks
```bash
python generate_embeddings.py
```
Reads data/train/output_masks/, creates normalized CLIP embeddings, saves compressed NPZ to data/train/embeddings/masked_clip_embeddings.npz

## 3.FAISS Index & Query
```bash
python image_search.py --mode build_index
python image_search.py --mode query --query-image /path/to/test.jpg --top-k 3
```
Build or query the index with a simple CLI over NPZ embeddings

## 4.Docker & Docker Compose
Ensure Docker is running, then:
```bash
docker-compose up --build -d
```
API → http://localhost:8000
Volumes mount your uploads, masks, embeddings, index, and mapping for persistence 

To stop and remove containers:
```bash
docker-compose down
```



## Run FastAPI inference server

```
uvicorn app:app --host 0.0.0.0 --port 8000
```
Open the interactive docs at:
Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
