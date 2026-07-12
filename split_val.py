import os
import random
import shutil

# -----------------------------
# Config
# -----------------------------
DATASET_DIR = "dataset"
VAL_RATIO = 0.1   # 10% of train images go to val
SEED = 42

img_train_dir = os.path.join(DATASET_DIR, "images", "train")
lbl_train_dir = os.path.join(DATASET_DIR, "labels", "train")
img_val_dir = os.path.join(DATASET_DIR, "images", "val")
lbl_val_dir = os.path.join(DATASET_DIR, "labels", "val")

os.makedirs(img_val_dir, exist_ok=True)
os.makedirs(lbl_val_dir, exist_ok=True)

# -----------------------------
# Collect images that have a matching label file
# -----------------------------
images = [f for f in os.listdir(img_train_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.seed(SEED)
random.shuffle(images)

val_count = int(len(images) * VAL_RATIO)
val_images = images[:val_count]

print(f"Total train images: {len(images)}")
print(f"Moving {val_count} images to val ({VAL_RATIO*100:.0f}%)")

moved = 0
skipped = 0

for img_name in val_images:
    base_name = os.path.splitext(img_name)[0]
    label_name = base_name + ".txt"

    src_img = os.path.join(img_train_dir, img_name)
    src_lbl = os.path.join(lbl_train_dir, label_name)

    dst_img = os.path.join(img_val_dir, img_name)
    dst_lbl = os.path.join(lbl_val_dir, label_name)

    if not os.path.exists(src_lbl):
        # No label file (e.g. image had no valid annotations) - skip it
        skipped += 1
        continue

    shutil.move(src_img, dst_img)
    shutil.move(src_lbl, dst_lbl)
    moved += 1

print(f" Moved {moved} image/label pairs to val")
if skipped:
    print(f" Skipped {skipped} images with no matching label file")