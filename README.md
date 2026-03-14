# CIFAR-10 Vision API

A CNN image classifier trained on the CIFAR-10 dataset, served as a REST API using FastAPI. Send any image and get back a predicted class label.

---


## Project Structure

```
cifar10-vision-api/
├── model.py          # CNN architecture
├── train.py          # Training script
├── predict.py        # Inference logic
├── main.py           # FastAPI app
├── saved_model.pth   # Trained model weights
└── requirements.txt
```

---

## Model Architecture

A custom CNN built with PyTorch:

| Layer        | Details                        |
|--------------|-------------------------------|
| Conv2d 1     | 3 → 32 filters, kernel 3×3   |
| MaxPool      | 2×2                           |
| Conv2d 2     | 32 → 64 filters, kernel 3×3  |
| MaxPool      | 2×2                           |
| Flatten      | 64 × 6 × 6 = 2304            |
| Linear 1     | 2304 → 128                    |
| Linear 2     | 128 → 10 (output classes)    |

**Dataset:** CIFAR-10 — 60,000 images (50k train / 10k test), 32×32 RGB  
**Training:** 25 epochs, Adam optimizer (lr=0.001), CrossEntropyLoss, batch size 32

---

## Classes

| Label | Class       |
|-------|-------------|
| 0     | Airplane    |
| 1     | Automobile  |
| 2     | Bird        |
| 3     | Cat         |
| 4     | Deer        |
| 5     | Dog         |
| 6     | Frog        |
| 7     | Horse       |
| 8     | Ship        |
| 9     | Truck       |

---

## Setup & Installation

**1. Clone the repo**
```bash
git clone https://github.com/your-username/cifar10-vision-api.git
cd cifar10-vision-api
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model** *(or skip if using the provided `saved_model.pth`)*
```bash
python train.py
```

**4. Start the API**
```bash
uvicorn main:app --reload
```

API will be live at `http://127.0.0.1:8000`

---

## API Usage

### Health check
```bash
GET http://127.0.0.1:8000/
# Response: { "message": "ML API running" }
```

### Predict
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@your_image.jpg"
# Response: { "prediction": "cat" }
```

Or use the auto-generated Swagger UI at `http://127.0.0.1:8000/docs`

---

## Requirements

```
torch
torchvision
fastapi
uvicorn
pillow
```


## Tech Stack

- **PyTorch** — model training and inference
- **FastAPI** — REST API framework
- **Pillow** — image loading
- **Uvicorn** — ASGI server

---

## Future Improvements

- [ ] Add Normalize() transform for better accuracy
- [ ] Add confidence score to API response
- [ ] Dockerize the application
