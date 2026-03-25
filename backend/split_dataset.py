import os
import random
import shutil

source_dir = "processed_dataset"
target_dir = "dataset_final"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

classes = ["mild", "moderate", "severe"]

for split in ["train", "val", "test"]:
    for c in classes:
        os.makedirs(os.path.join(target_dir, split, c), exist_ok=True)

for c in classes:

    class_path = os.path.join(source_dir, c)
    images = os.listdir(class_path)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for img in train_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(target_dir, "train", c, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(target_dir, "val", c, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(target_dir, "test", c, img)
        )

    print(f"{c} split complete.")