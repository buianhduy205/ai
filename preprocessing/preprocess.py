
# ============================================================
# PREPROCESSING DATASET - FRUIT QUALITY CLASSIFICATION
#
# Chức năng:
# 1. Đọc dataset FruitQuality1
# 2. Kiểm tra ảnh hợp lệ
# 3. Chia Train / Validation / Test (70/15/15)
# 4. Chia riêng từng lớp Fresh / Rotten
# 5. Đảm bảo mỗi lớp xuất hiện trong cả 3 tập
# 6. Giữ nguyên dataset cũ nếu quá trình xử lý thất bại
# 7. Thống kê dữ liệu sau khi chia
# ============================================================

import random
import shutil
from pathlib import Path

from PIL import Image


# ============================================================
# 1. CẤU HÌNH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "dataset" / "fruitquality1" / "train"

OUTPUT_DIR = BASE_DIR / "dataset" / "processed_dataset"

TEMP_DIR = BASE_DIR / "dataset" / "processed_dataset_temp"

BACKUP_DIR = BASE_DIR / "dataset" / "processed_dataset_backup"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

VALID_EXTENSIONS = [
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
]

CONDITIONS = ["fresh", "rotten"]

SPLITS = ["train", "val", "test"]


# ============================================================
# 2. KIỂM TRA CẤU HÌNH
# ============================================================

if not abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-9:
    raise ValueError("Tổng tỷ lệ chia phải bằng 1!")

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dataset nguồn: {SOURCE_DIR}"
    )

if TEMP_DIR.exists() or BACKUP_DIR.exists():
    raise RuntimeError(
        "Phát hiện thư mục tạm hoặc backup từ lần chạy trước. "
        "Hãy kiểm tra và xử lý trước khi chạy lại."
    )

rng = random.Random(RANDOM_SEED)


# ============================================================
# 3. HÀM KIỂM TRA ẢNH
# ============================================================

def is_valid_image(image_path):

    try:
        with Image.open(image_path) as image:
            image.verify()

        return True

    except Exception:
        return False


# ============================================================
# 4. HÀM CHIA DỮ LIỆU
# ============================================================

def split_images(images):

    total = len(images)

    if total < 3:
        raise ValueError(
            f"Lớp chỉ có {total} ảnh. "
            "Cần ít nhất 3 ảnh để chia thành 3 tập."
        )

    images = images.copy()

    rng.shuffle(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    # Đảm bảo mỗi tập có ít nhất 1 ảnh
    train_count = max(1, train_count)
    val_count = max(1, val_count)

    # Dành ít nhất 1 ảnh cho Test
    if train_count + val_count >= total:
        train_count = total - 2
        val_count = 1

    train_images = images[:train_count]

    val_images = images[
        train_count:train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]

    return {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }


# ============================================================
# 5. ĐỌC VÀ KIỂM TRA DATASET NGUỒN
# ============================================================

print("=" * 65)
print("PREPROCESSING FRUIT QUALITY DATASET")
print("=" * 65)

print(f"\nDataset nguồn: {SOURCE_DIR}")

fruit_folders = sorted([
    folder for folder in SOURCE_DIR.iterdir()
    if folder.is_dir()
])

if not fruit_folders:
    raise RuntimeError("Không tìm thấy loại trái cây nào!")

print(f"\nSố loại trái cây: {len(fruit_folders)}")

dataset_information = {}

total_valid = 0
total_invalid = 0

for fruit_folder in fruit_folders:

    fruit_name = fruit_folder.name

    for condition in CONDITIONS:

        condition_folder = fruit_folder / condition

        if not condition_folder.exists():
            raise FileNotFoundError(
                f"Thiếu thư mục: {condition_folder}"
            )

        images = [
            file for file in condition_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        ]

        valid_images = []
        invalid_images = []

        for image_path in images:

            if is_valid_image(image_path):
                valid_images.append(image_path)
            else:
                invalid_images.append(image_path)

        if len(valid_images) < 3:
            raise ValueError(
                f"Lớp {fruit_name}_{condition} "
                f"chỉ có {len(valid_images)} ảnh hợp lệ. "
                "Cần ít nhất 3 ảnh."
            )

        class_name = f"{fruit_name}_{condition}"

        dataset_information[class_name] = valid_images

        total_valid += len(valid_images)
        total_invalid += len(invalid_images)

        print(
            f"{class_name:<30} "
            f"Hợp lệ: {len(valid_images):>5} | "
            f"Lỗi: {len(invalid_images):>4}"
        )


# ============================================================
# 6. CHIA DỮ LIỆU
# ============================================================

print("\nĐang chia dữ liệu...")

split_information = {
    "train": {},
    "val": {},
    "test": {}
}

for class_name, images in dataset_information.items():

    split_result = split_images(images)

    for split in SPLITS:

        split_information[split][class_name] = (
            split_result[split]
        )


# ============================================================
# 7. TẠO DATASET TẠM
# ============================================================

TEMP_DIR.mkdir(parents=True, exist_ok=False)

try:

    for split in SPLITS:

        for class_name in dataset_information:

            output_folder = (
                TEMP_DIR / split / class_name
            )

            output_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            images = split_information[split][class_name]

            for image_path in images:

                shutil.copy2(
                    image_path,
                    output_folder / image_path.name
                )

    # Kiểm tra lại số ảnh sau khi copy
    for split in SPLITS:

        for class_name in dataset_information:

            output_folder = TEMP_DIR / split / class_name

            copied_images = [
                file for file in output_folder.iterdir()
                if file.is_file()
            ]

            expected_count = len(
                split_information[split][class_name]
            )

            if len(copied_images) != expected_count:
                raise RuntimeError(
                    f"Lỗi copy dữ liệu: {class_name}/{split}"
                )


    # ========================================================
    # 8. KIỂM TRA KHÔNG TRÙNG ẢNH GIỮA CÁC TẬP
    # ========================================================

    for class_name in dataset_information:

        train_set = {
            image.name
            for image in split_information["train"][class_name]
        }

        val_set = {
            image.name
            for image in split_information["val"][class_name]
        }

        test_set = {
            image.name
            for image in split_information["test"][class_name]
        }

        if train_set & val_set:
            raise RuntimeError(
                f"Ảnh trùng Train/Val: {class_name}"
            )

        if train_set & test_set:
            raise RuntimeError(
                f"Ảnh trùng Train/Test: {class_name}"
            )

        if val_set & test_set:
            raise RuntimeError(
                f"Ảnh trùng Val/Test: {class_name}"
            )


    # ========================================================
    # 9. THAY DATASET CŨ AN TOÀN
    # ========================================================

    if OUTPUT_DIR.exists():

        OUTPUT_DIR.rename(BACKUP_DIR)

    try:

        TEMP_DIR.rename(OUTPUT_DIR)

    except Exception:

        if BACKUP_DIR.exists():
            BACKUP_DIR.rename(OUTPUT_DIR)

        raise


    # Xóa backup sau khi dataset mới đã thay thế thành công
    if BACKUP_DIR.exists():

        shutil.rmtree(BACKUP_DIR)


except Exception:

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    raise


# ============================================================
# 10. THỐNG KÊ KẾT QUẢ
# ============================================================

print("\n" + "=" * 65)
print("THỐNG KÊ DATASET SAU PREPROCESSING")
print("=" * 65)

totals = {
    "train": 0,
    "val": 0,
    "test": 0
}

for split in SPLITS:

    print(f"\n[{split.upper()}]")

    for class_name in dataset_information:

        count = len(
            split_information[split][class_name]
        )

        totals[split] += count

        print(f"{class_name:<30} : {count:>5} ảnh")

    print(
        f"Tổng {split}: {totals[split]} ảnh"
    )


# ============================================================
# 11. TỔNG KẾT
# ============================================================

print("\n" + "=" * 65)
print("PREPROCESSING HOÀN TẤT!")
print("=" * 65)

print(f"Số loại trái cây: {len(fruit_folders)}")

print(f"Số lớp phân loại: {len(dataset_information)}")

print(f"Tổng ảnh hợp lệ: {total_valid}")

print(f"Tổng ảnh lỗi: {total_invalid}")

print(f"Train: {totals['train']} ảnh")

print(f"Validation: {totals['val']} ảnh")

print(f"Test: {totals['test']} ảnh")

print(
    f"Tổng sau chia: "
    f"{sum(totals.values())} ảnh"
)

print(f"\nDataset mới: {OUTPUT_DIR}")

print("\nCấu trúc:")

print("processed_dataset/")
print("├── train/")
print("├── val/")
print("└── test/")

print("\nDataset cũ chỉ được thay thế sau khi xử lý thành công.")
