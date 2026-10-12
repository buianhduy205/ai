# File thực hiện quá trình TRAIN model.
# Đọc dataset -> đưa ảnh vào model -> tính loss -> cập nhật trọng số.
# Sau khi train xong, lưu model tốt nhất để sử dụng cho dự đoán.
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ==========================================
# 1. ĐƯỜNG DẪN PROJECT
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "processed_dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CẤU HÌNH
# ==========================================

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("TRAIN MOBILENETV2 - FRUIT QUALITY")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_DIR}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")


# ==========================================
# 3. KIỂM TRA DATASET
# ==========================================

train_dir = DATASET_DIR / "train"
val_dir = DATASET_DIR / "val"

if not train_dir.exists():
    print("\nERROR: Không tìm thấy thư mục train!")
    print(train_dir)
    sys.exit()

if not val_dir.exists():
    print("\nERROR: Không tìm thấy thư mục val!")
    print(val_dir)
    sys.exit()


# ==========================================
# 4. PREPROCESSING
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 5. LOAD DATASET
# ==========================================

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=val_transform
)


# ==========================================
# 6. THÔNG TIN DATASET
# ==========================================

class_names = train_dataset.classes
num_classes = len(class_names)

print("\nSố lớp:", num_classes)

print("\nDanh sách lớp:")

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")

print("\nSố ảnh train:", len(train_dataset))
print("Số ảnh validation:", len(val_dataset))


# ==========================================
# 7. DATA LOADER
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 8. TẠO MOBILENETV2
# ==========================================

print("\nĐang tạo MobileNetV2...")

model = models.mobilenet_v2(weights="DEFAULT")


# Thay lớp classifier cuối
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(DEVICE)


# ==========================================
# 9. LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 10. TRAIN
# ==========================================

best_val_accuracy = 0.0

model_path = MODEL_DIR / "mobilenetv2_fruitquality_best.pth"


for epoch in range(EPOCHS):

    print("\n" + "=" * 60)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    # --------------------------
    # TRAIN
    # --------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = (
        train_correct / train_total
    ) * 100

    average_train_loss = (
        train_loss / len(train_loader)
    )


    # --------------------------
    # VALIDATION
    # --------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = (
        val_correct / val_total
    ) * 100

    average_val_loss = (
        val_loss / len(val_loader)
    )


    # --------------------------
    # IN KẾT QUẢ
    # --------------------------

    print(
        f"Train Loss: {average_train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {average_val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_accuracy:.2f}%"
    )


    # --------------------------
    # SAVE BEST MODEL
    # --------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "image_size": IMAGE_SIZE
            },
            model_path
        )

        print(
            f"Đã lưu model tốt nhất: {model_path}"
        )


# ==========================================
# 11. HOÀN TẤT
# ==========================================

print("\n" + "=" * 60)
print("TRAIN HOÀN TẤT")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Model được lưu tại:\n{model_path}"
)