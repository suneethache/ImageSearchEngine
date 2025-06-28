
from transformers import CLIPProcessor, CLIPModel

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("loaded")


from fastapi import FastAPI
app = FastAPI()

clip_processor = None

from fastapi import FastAPI
app = FastAPI()