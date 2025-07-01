import io
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import io
import os
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Welcome to the Image Similarity Finder API"}
# Load model once on startup
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def image_to_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings.cpu().numpy()

def load_folder_embeddings(folder_path: str):
    embeddings = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            filepath = os.path.join(folder_path, filename)
            try:
                image = Image.open(filepath).convert("RGB")
                emb = image_to_embedding(image)
                embeddings.append(emb)
                filenames.append(filename)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    if embeddings:
        embeddings = torch.tensor(embeddings).squeeze(1).numpy()
    else:
        embeddings = []
    return embeddings, filenames

@app.post("/find_similar/")
async def find_similar(image: UploadFile = File(...), folder_path: str = Form(...)):
    # Read uploaded image
    img_bytes = await image.read()
    query_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Compute embedding for query image
    query_emb = image_to_embedding(query_img)

    # Load folder embeddings
    if not os.path.exists(folder_path):
        return JSONResponse(status_code=400, content={"error": "Folder path does not exist"})

    folder_embeddings, filenames = load_folder_embeddings(folder_path)

    if len(folder_embeddings) == 0:
        return JSONResponse(status_code=400, content={"error": "No images found in folder"})

    # Compute cosine similarities
    sims = cosine_similarity(query_emb, folder_embeddings)[0]

    # Find best match
    best_idx = sims.argmax()
    best_filename = filenames[best_idx]
    best_score = float(sims[best_idx])

    return {
        "best_match_filename": best_filename,
        "similarity_score": best_score
    }
