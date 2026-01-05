from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from app.model import load_model
from app.utils import preprocess_image
from app.inference import predict

app = FastAPI(title="Image Classification API")

model = load_model("weights/weights.pth")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    x = preprocess_image(img)
    result = predict(model, x)

    return result
