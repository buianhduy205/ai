# Tiền xử lý dataset trước khi đưa vào quá trình train.
# Thực hiện các bước như chia dữ liệu, resize ảnh, augmentation...
# Đầu ra là dataset đã được chuẩn bị để huấn luyện model.
import os
import shutil
import random
from pathlib import Path

# ==============================
# 1. CẤU HÌNH
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "dataset" / "fruitquality1" / "train"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed_dataset"

# Tỷ lệ chia dữ liệu
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Đảm bảo kết quả chia giống nhau mỗi lần chạy
random.seed(42)


# ==============================
# 2. KIỂM TRA TỶ LỆ
# ==============================

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0


# ==============================
# 3. XÓA DATASET CŨ NẾU CÓ
# ==============================

if OUTPUT_DIR.exists():
    print("Đang xóa processed_dataset cũ...")
    shutil.rmtree(OUTPUT_DIR)


# ==============================
# 4. TẠO THƯ MỤC
# ==============================

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)


# ==============================
# 5. ĐỌC CÁC LOẠI TRÁI CÂY
# ==============================

fruit_classes = [
    folder for folder in SOURCE_DIR.iterdir()
    if folder.is_dir()
]

print(f"\nSố loại trái cây: {len(fruit_classes)}")


# ==============================
# 6. CHIA DỮ LIỆU
# ==============================

total_images = 0

for fruit_folder in sorted(fruit_classes):

    fruit_name = fruit_folder.name

    print(f"\nĐang xử lý: {fruit_name}")

    # Fresh và Rotten
    for condition in ["fresh", "rotten"]:

        condition_folder = fruit_folder / condition

        if not condition_folder.exists():
            print(f"  Không tìm thấy: {condition_folder}")
            continue

        # Lấy tất cả file ảnh
        images = [
            file for file in condition_folder.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ]

        # Trộn ảnh
        random.shuffle(images)

        # Tính số lượng
        total = len(images)

        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        print(
            f"  {condition}: "
            f"tổng={total}, "
            f"train={len(train_images)}, "
            f"val={len(val_images)}, "
            f"test={len(test_images)}"
        )

        # ==========================
        # Tạo thư mục output
        # ==========================

        for split, split_images in [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images)
        ]:

            output_folder = (
                OUTPUT_DIR
                / split
                / f"{fruit_name}_{condition}"
            )

            output_folder.mkdir(parents=True, exist_ok=True)

            # Copy ảnh
            for image in split_images:
                shutil.copy2(
                    image,
                    output_folder / image.name
                )

        total_images += total


# ==============================
# 7. KẾT QUẢ
# ==============================

print("\n" + "=" * 50)
print("HOÀN TẤT PREPROCESSING")
print("=" * 50)

print(f"Tổng số ảnh: {total_images}")

print("\nDataset mới được tạo tại:")
print(OUTPUT_DIR.resolve())

print("\nCấu trúc:")
print("processed_dataset/")
print("├── train/")
print("├── val/")
print("└── test/")