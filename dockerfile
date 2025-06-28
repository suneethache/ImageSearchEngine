FROM python:3.10-slim-bullseye@sha256:<digest>

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Create data directories
RUN mkdir -p uploads masks data/train/embeddings data/train/index data/train/mapping

# Expose port for FastAPI
EXPOSE 8000

# Entrypoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]