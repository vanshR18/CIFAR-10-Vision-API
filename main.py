from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from predict import predict_image

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ML API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents))

    prediction = predict_image(image)

    return {"prediction": prediction}