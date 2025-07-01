from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import io
import os

import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Load model and processor once on startup
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Put model in eval mode and to CPU or GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def get_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    # Normalize embedding
    embeddings /= embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings.cpu().numpy()

@app.post("/find-similar/")
async def find_similar_image(
    query_image: UploadFile = File(...),
    folder_path: str = Form(...)
):
    # Check folder path exists
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Folder path does not exist on server.")

    # Load query image
    try:
        query_img_bytes = await query_image.read()
        query_img = Image.open(io.BytesIO(query_img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid query image.")

    query_embedding = get_image_embedding(query_img)

    # Scan folder for images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]

    if not image_files:
        raise HTTPException(status_code=400, detail="No images found in folder.")

    similarities = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            emb = get_image_embedding(img)
            sim = cosine_similarity(query_embedding, emb)[0][0]
            similarities.append((img_file, sim))
        except Exception:
            # skip unreadable images
            continue

    if not similarities:
        raise HTTPException(status_code=400, detail="No valid images found to compare.")

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top 3 similar images with similarity scores
    top_results = [{"filename": fname, "similarity": float(sim)} for fname, sim in similarities[:3]]

    return JSONResponse(content={"results": top_results})
