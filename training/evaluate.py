# File đánh giá model sau khi đã train.
# Dùng tập test để kiểm tra model dự đoán chính xác đến đâu.
# Tạo các chỉ số đánh giá và các kết quả như classification report,
# confusion matrix và accuracy.
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. ĐƯỜNG DẪN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "dataset" / "processed_dataset" / "test"

MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_fruitquality_best.pth"

RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. CẤU HÌNH
# ==========================================

IMAGE_SIZE = 128
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("ĐÁNH GIÁ MOBILENETV2")
print("=" * 60)

print("Device:", DEVICE)
print("Test:", TEST_DIR)
print("Model:", MODEL_PATH)


# ==========================================
# 3. PREPROCESSING
# ==========================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. LOAD TEST DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes

print("\nSố class:", len(class_names))
print("Số ảnh test:", len(test_dataset))


# ==========================================
# 5. LOAD MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

num_classes = len(class_names)

model = models.mobilenet_v2(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ==========================================
# 6. DỰ ĐOÁN
# ==========================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ==========================================
# 7. TÍNH METRICS
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ==========================================
# 8. IN KẾT QUẢ
# ==========================================

print("\n" + "=" * 60)
print("KẾT QUẢ ĐÁNH GIÁ")
print("=" * 60)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-score : {f1 * 100:.2f}%")


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    zero_division=0
)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)

with open(
    RESULT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(20, 18))

sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - MobileNetV2")

plt.xticks(
    rotation=90,
    fontsize=7
)

plt.yticks(
    rotation=0,
    fontsize=7
)

plt.tight_layout()

cm_path = RESULT_DIR / "confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()


# ==========================================
# 11. LƯU METRICS
# ==========================================

with open(
    RESULT_DIR / "metrics.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    f.write(
        f"Recall: {recall * 100:.2f}%\n"
    )

    f.write(
        f"F1-score: {f1 * 100:.2f}%\n"
    )


print("\nĐã lưu kết quả tại:")
print(RESULT_DIR)

print("\nConfusion Matrix:")
print(cm_path)

print("\nĐÁNH GIÁ HOÀN TẤT!")