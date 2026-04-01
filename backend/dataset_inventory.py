import os
# path to raw datasets
base_path = "raw_datasets"
print("\nScanning Datasets...\n")
for dataset in os.listdir(base_path):
    dataset_path = os.path.join(base_path,dataset)
    if(os.path.isdir(dataset_path)):
        print(f"\nDataset: {dataset}")
        print("-" * 40)

        for root, dirs, files in os.walk(dataset_path):

            image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            if len(image_files) > 0:
                print(f"Folder: {root}")
                print(f"Number of images: {len(image_files)}\n")