# File chính của AI Service.
# Dùng để khởi chạy API và kết nối model AI với ứng dụng.
# Nhận ảnh đầu vào -> đưa vào model -> trả về kết quả dự đoán.

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException


# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_fruitquality_best.pth"


# ============================================================
# 2. THIẾT LẬP
# ============================================================

IMAGE_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]


model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# 4. XỬ LÝ ẢNH
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. FASTAPI
# ============================================================

app = FastAPI(
    title="Fruit Quality AI",
    description="API nhận diện trái cây và phân loại Fresh/Rotten bằng MobileNetV2",
    version="1.0"
)


# ============================================================
# 6. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "num_classes": num_classes,
        "device": str(DEVICE)
    }


# ============================================================
# 7. PREDICT
# ============================================================

@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    # Kiểm tra file ảnh
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File tải lên phải là hình ảnh."
        )

    try:
        # Đọc ảnh
        image_bytes = await image.read()

        img = Image.open(
            __import__("io").BytesIO(image_bytes)
        ).convert("RGB")

        # Preprocess
        input_tensor = transform(img)

        input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(DEVICE)

        # Dự đoán
        with torch.no_grad():
            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        # Lấy class
        predicted_index = predicted.item()

        predicted_class = class_names[predicted_index]

        confidence_value = confidence.item()

        # Tách tên trái cây và trạng thái
        parts = predicted_class.rsplit("_", 1)

        if len(parts) == 2:
            fruit = parts[0]
            condition = parts[1]
        else:
            fruit = predicted_class
            condition = "unknown"

        return {
            "filename": image.filename,
            "prediction": predicted_class,
            "fruit": fruit,
            "condition": condition,
            "confidence": round(
                confidence_value * 100,
                2
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xử lý ảnh: {str(e)}"
        )
        