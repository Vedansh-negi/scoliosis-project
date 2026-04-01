import os
from PIL import Image

raw_path = "raw_datasets"
output_path = "processed_dataset"

# create output folders
classes = ["mild", "moderate", "severe"]

for c in classes:
    os.makedirs(os.path.join(output_path, c), exist_ok=True)

img_counter = 0


def process_image(src_path, label):

    global img_counter

    try:
        img = Image.open(src_path)
        img = img.convert("RGB")
        img = img.resize((224,224))

        img_counter += 1

        save_path = os.path.join(output_path, label, f"img_{img_counter}.jpg")

        img.save(save_path)

    except:
        pass


print("\nProcessing XKT dataset...")

xkt_path = os.path.join(raw_path, "xkt", "224","224")

mapping = {
    "Normal": "mild",
    "Scol": "moderate",
    "Spond": "severe"
}

for folder in mapping:

    folder_path = os.path.join(xkt_path, folder)

    for img in os.listdir(folder_path):

        img_path = os.path.join(folder_path, img)

        process_image(img_path, mapping[folder])


print("Processing curvature dataset...")

curv_train = os.path.join(raw_path, "datasetter", "training_set")
curv_test = os.path.join(raw_path, "datasetter", "test_set")

mapping2 = {
    "c": "mild",
    "s": "moderate"
}

for dataset in [curv_train, curv_test]:

    for folder in mapping2:

        folder_path = os.path.join(dataset, folder)

        for img in os.listdir(folder_path):

            img_path = os.path.join(folder_path, img)

            process_image(img_path, mapping2[folder])


print("Processing YOLO dataset...")

yolo_path = os.path.join(raw_path, "scoliosis2")

for split in ["train", "valid", "test"]:

    img_folder = os.path.join(yolo_path, split, "images")

    for img in os.listdir(img_folder):

        img_path = os.path.join(img_folder, img)

        process_image(img_path, "moderate")


print("\nDataset merge complete.")
print(f"Total images processed: {img_counter}")