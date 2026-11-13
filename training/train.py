
# ============================================================
# TRAIN MOBILENETV2 - FRUIT QUALITY CLASSIFICATION
# Version: mobilenetv2-fruit-v4
# ============================================================

import sys
import csv
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models


# ============================================================
# 1. DUONG DAN PROJECT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "processed_dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "ai" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CAU HINH HUAN LUYEN
# ============================================================

MODEL_VERSION = "mobilenetv2-fruit-v4"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = MODEL_DIR / f"{MODEL_VERSION}.pth"
JSON_PATH = RESULT_DIR / f"{MODEL_VERSION}.json"
CSV_PATH = RESULT_DIR / f"{MODEL_VERSION}.csv"


# ============================================================
# 3. RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 4. IN THONG TIN
# ============================================================

print("=" * 65)
print("MOBILENETV2 - FRUIT QUALITY CLASSIFICATION")
print("=" * 65)

print(f"Model version : {MODEL_VERSION}")
print(f"Device        : {DEVICE}")
print(f"Dataset       : {DATASET_DIR}")
print(f"Image size    : {IMAGE_SIZE}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Epochs        : {EPOCHS}")
print(f"Learning rate : {LEARNING_RATE}")


# ============================================================
# 5. KIEM TRA DU LIEU
# ============================================================

if not TRAIN_DIR.exists():
    sys.exit(f"Khong tim thay Train: {TRAIN_DIR}")

if not VAL_DIR.exists():
    sys.exit(f"Khong tim thay Validation: {VAL_DIR}")

if not any(TRAIN_DIR.iterdir()):
    sys.exit("Thu muc Train dang trong!")

if not any(VAL_DIR.iterdir()):
    sys.exit("Thu muc Validation dang trong!")

if MODEL_PATH.exists():
    sys.exit(
        f"Model {MODEL_VERSION} da ton tai! "
        "Hay doi MODEL_VERSION neu muon tao phien ban moi."
    )

if JSON_PATH.exists() or CSV_PATH.exists():
    sys.exit(
        "File ket qua cua phien ban nay da ton tai! "
        "Hay doi MODEL_VERSION."
    )


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

# Augmentation chi ap dung cho Train.
# Giup mo hinh lam quen voi nhieu goc chup,
# kich thuoc, anh sang va goc nhin khac nhau.

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.0),
        ratio=(0.85, 1.15)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=25
    ),

    transforms.RandomPerspective(
        distortion_scale=0.2,
        p=0.3
    ),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.05
    ),

    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=3)],
        p=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation khong dung augmentation ngau nhien.

val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. LOAD DATASET
# ============================================================

print("\nDang doc dataset...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

class_names = train_dataset.classes
num_classes = len(class_names)

if num_classes < 2:
    sys.exit("Dataset phai co it nhat 2 lop!")

if train_dataset.classes != val_dataset.classes:
    sys.exit(
        "Danh sach lop Train va Validation khong khop!"
    )

if len(train_dataset) == 0 or len(val_dataset) == 0:
    sys.exit("Train hoac Validation khong co anh!")

print(f"So lop: {num_classes}")
print(f"So anh Train: {len(train_dataset)}")
print(f"So anh Validation: {len(val_dataset)}")


# ============================================================
# 8. THONG KE SO ANH TUNG LOP
# ============================================================

train_targets = torch.tensor(train_dataset.targets)

class_counts = torch.bincount(
    train_targets,
    minlength=num_classes
)

print("\nTHONG KE DU LIEU TRAIN")
print("-" * 65)

for i, class_name in enumerate(class_names):

    count = class_counts[i].item()

    print(f"{class_name:<30}: {count:>6} anh")

    if count == 0:
        sys.exit(f"Lop {class_name} khong co anh Train!")


# ============================================================
# 9. WEIGHTED RANDOM SAMPLER
# ============================================================

print("\nDang tao WeightedRandomSampler...")

class_weights = 1.0 / class_counts.float()

sample_weights = class_weights[train_targets]

sampler_generator = torch.Generator()
sampler_generator.manual_seed(RANDOM_SEED)

sampler = WeightedRandomSampler(
    weights=sample_weights.double(),
    num_samples=len(train_dataset),
    replacement=True,
    generator=sampler_generator
)

print("Da tao bo lay mau can bang du lieu!")


# ============================================================
# 10. DATA LOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# 11. KHOI TAO MOBILENETV2
# ============================================================

print("\nDang tai MobileNetV2 pretrained...")

model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(DEVICE)


# ============================================================
# 12. LOSS FUNCTION VA OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 13. HAM TINH METRICS
# ============================================================

def calculate_metrics(cm):

    cm = cm.float()

    true_positive = torch.diag(cm)

    actual_positive = cm.sum(dim=1)
    predicted_positive = cm.sum(dim=0)

    precision_per_class = (
        true_positive / predicted_positive.clamp(min=1)
    )

    recall_per_class = (
        true_positive / actual_positive.clamp(min=1)
    )

    f1_per_class = (
        2 * precision_per_class * recall_per_class
        / (precision_per_class + recall_per_class).clamp(min=1e-12)
    )

    accuracy = (
        true_positive.sum() / cm.sum().clamp(min=1)
    ).item()

    precision = precision_per_class.mean().item()
    recall = recall_per_class.mean().item()
    f1_score = f1_per_class.mean().item()

    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1_score * 100
    }


# ============================================================
# 14. KHOI TAO BIEN
# ============================================================

best_val_accuracy = -1.0
best_val_loss = float("inf")

best_epoch = 0
best_metrics = None

history = []


# ============================================================
# 15. BAT DAU HUAN LUYEN
# ============================================================

for epoch in range(EPOCHS):

    print("\n" + "=" * 65)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 65)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

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

        batch_size = labels.size(0)

        train_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += batch_size

    average_train_loss = train_loss / train_total

    train_accuracy = (
        train_correct / train_total
    ) * 100

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_total = 0

    confusion_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long
    )

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)

            val_loss += loss.item() * batch_size
            val_total += batch_size

            encoded = (
                labels * num_classes + predictions
            )

            confusion_matrix += torch.bincount(
                encoded.cpu(),
                minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    average_val_loss = val_loss / val_total

    metrics = calculate_metrics(confusion_matrix)

    val_accuracy = metrics["accuracy"]

    # --------------------------------------------------------
    # IN KET QUA
    # --------------------------------------------------------

    print(f"Train Loss     : {average_train_loss:.4f}")
    print(f"Train Accuracy : {train_accuracy:.2f}%")

    print(f"Val Loss       : {average_val_loss:.4f}")
    print(f"Val Accuracy   : {val_accuracy:.2f}%")

    print(f"Precision      : {metrics['precision']:.2f}%")
    print(f"Recall         : {metrics['recall']:.2f}%")
    print(f"F1-score       : {metrics['f1_score']:.2f}%")

    # --------------------------------------------------------
    # LUU LICH SU EPOCH
    # --------------------------------------------------------

    epoch_result = {
        "epoch": epoch + 1,
        "train_loss": round(average_train_loss, 6),
        "train_accuracy": round(train_accuracy, 4),
        "val_loss": round(average_val_loss, 6),
        "val_accuracy": round(val_accuracy, 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1_score": round(metrics["f1_score"], 4)
    }

    history.append(epoch_result)

    # --------------------------------------------------------
    # LUU MODEL TOT NHAT
    # --------------------------------------------------------

    is_better = (
        val_accuracy > best_val_accuracy
        or (
            val_accuracy == best_val_accuracy
            and average_val_loss < best_val_loss
        )
    )

    if is_better:

        best_val_accuracy = val_accuracy
        best_val_loss = average_val_loss
        best_epoch = epoch + 1
        best_metrics = metrics.copy()

        checkpoint = {
            "model_version": MODEL_VERSION,
            "model_name": "MobileNetV2",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "num_classes": num_classes,
            "image_size": IMAGE_SIZE,
            "epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "metrics": best_metrics,
            "created_at": datetime.now().isoformat()
        }

        torch.save(checkpoint, MODEL_PATH)

        print("\nDA LUU MODEL TOT NHAT!")
        print(f"Model: {MODEL_PATH}")
        print(f"Best Accuracy: {best_val_accuracy:.2f}%")


# ============================================================
# 16. LUU LICH SU CSV
# ============================================================

csv_fields = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "precision",
    "recall",
    "f1_score"
]

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=csv_fields
    )

    writer.writeheader()
    writer.writerows(history)


# ============================================================
# 17. LUU KET QUA JSON
# ============================================================

results = {
    "model_version": MODEL_VERSION,
    "model_name": "MobileNetV2",
    "dataset": str(DATASET_DIR),
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "random_seed": RANDOM_SEED,
    "device": str(DEVICE),
    "num_classes": num_classes,
    "class_names": class_names,
    "train_images": len(train_dataset),
    "validation_images": len(val_dataset),

    "class_distribution": {
        name: int(class_counts[i].item())
        for i, name in enumerate(class_names)
    },

    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_accuracy,
    "best_val_loss": best_val_loss,
    "best_metrics": best_metrics,
    "history": history,
    "created_at": datetime.now().isoformat()
}

with open(
    JSON_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 18. KET THUC
# ============================================================

print("\n" + "=" * 65)
print("TRAIN HOAN TAT!")
print("=" * 65)

print(f"Model version : {MODEL_VERSION}")
print(f"Best epoch    : {best_epoch}")
print(f"Best Accuracy : {best_val_accuracy:.2f}%")
print(f"Best Loss     : {best_val_loss:.4f}")

print("\nCac file da luu:")
print(f"Model : {MODEL_PATH}")
print(f"JSON  : {JSON_PATH}")
print(f"CSV   : {CSV_PATH}")

print("\nCac model phien ban cu duoc giu nguyen.")
print("=" * 65)
