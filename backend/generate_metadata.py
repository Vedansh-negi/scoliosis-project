import os
import csv
import random

dataset_dir = "dataset_final"

mild_templates = [
"Mild spinal curvature detected with Cobb angle between 10 and 20 degrees.",
"Minor spinal deviation observed in the thoracic region.",
"Patient exhibits early stage scoliosis with minimal curvature.",
"Slight curvature of the spine visible in X-ray imaging.",
"Mild scoliosis detected with minimal structural deformity."
]

moderate_templates = [
"Moderate spinal curvature observed with Cobb angle around 25–40 degrees.",
"Noticeable scoliosis present in thoracic spine region.",
"Patient reports moderate back discomfort with visible spinal deviation.",
"Moderate scoliosis detected with clear vertebral misalignment.",
"Spinal curvature indicates intermediate scoliosis progression."
]

severe_templates = [
"Severe spinal curvature detected with Cobb angle exceeding 45 degrees.",
"Advanced scoliosis observed with significant vertebral rotation.",
"Severe deformity of the spine visible in radiographic imaging.",
"Pronounced spinal curvature indicating advanced scoliosis stage.",
"Critical scoliosis condition with major spinal displacement."
]

template_map = {
"mild": mild_templates,
"moderate": moderate_templates,
"severe": severe_templates
}

rows = []

for split in ["train","val","test"]:
    split_path = os.path.join(dataset_dir, split)

    for label in os.listdir(split_path):

        class_path = os.path.join(split_path, label)

        for img in os.listdir(class_path):

            note = random.choice(template_map[label])

            rows.append([
                os.path.join(split,label,img),
                label,
                note
            ])

with open("dataset_metadata.csv","w",newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["image_path","label","clinical_note"])

    writer.writerows(rows)

print("Metadata file created.")