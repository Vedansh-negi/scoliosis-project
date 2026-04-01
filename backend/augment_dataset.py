import os
import cv2
import random

dataset_path = "processed_dataset"

target_size = 2000


def augment_image(image):

    choice = random.choice(["flip", "rotate", "brightness"])

    if choice == "flip":
        return cv2.flip(image, 1)

    if choice == "rotate":
        rows, cols, _ = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), random.randint(-15,15), 1)
        return cv2.warpAffine(image, M, (cols, rows))

    if choice == "brightness":
        value = random.randint(-30,30)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = cv2.add(hsv[:,:,2], value)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


for label in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, label)

    images = os.listdir(class_path)

    print(f"\nProcessing class: {label}")
    print(f"Current images: {len(images)}")

    while len(images) < target_size:

        img_name = random.choice(images)

        img_path = os.path.join(class_path, img_name)

        image = cv2.imread(img_path)

        aug_img = augment_image(image)

        new_name = f"aug_{random.randint(100000,999999)}.jpg"

        cv2.imwrite(os.path.join(class_path,new_name), aug_img)

        images.append(new_name)

    print(f"Final images: {len(images)}")