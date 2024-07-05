import argparse
from collections import Counter
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random


def _augment_and_save_img(augmenter, dataset_dict):
    for label, images in dataset_dict.items():
        num_images = len(images)
        if num_images < max_class_size:
            for i in range(max_class_size - num_images):
                image_path = random.choice(images)
                image = cv2.imread(image_path)
                augmented_image = augmenter(image=image)['image']
                base_name = os.path.basename(image_path).split('.')[0]
                new_file_name = f"{base_name}_aug_{i}.jpg"
                # Define the new file path within the appropriate class folder
                class_folder = os.path.dirname(image_path)
                new_file_path = os.path.join(class_folder, new_file_name)
                cv2.imwrite(new_file_path, augmented_image)


def _extract_img_labels(metadata_path, img_dir):
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)

    # Extract images and labels
    images = []
    labels = []

    for index, row in metadata_df.iterrows():
        file_name = row['image_id'] + '.jpg'
        img_path = os.path.join(img_dir, row['dx'], file_name).replace("/", "\\")
        images.append(img_path)
        labels.append(row['dx'])  # Assuming 'dx' column contains the class labels

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Oversampling the train split of HAM10000 dataset for a more balanced dataset.')
    parser.add_argument('--metadata_file', type=str, required=True, help='Path to the metadata CSV file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the images.')

    args = parser.parse_args()

    print("Loading train split info...")
    train_img, train_labels = _extract_img_labels(args.metadata_file, args.image_dir)
    # Organize images by class
    class_images = defaultdict(list)
    for img_path, label in zip(train_img, train_labels):
        class_images[label].append(img_path)

    max_class_size = max(len(images) for images in class_images.values())
    print("The max class size is {}".format(max_class_size))
    print("Setting up the augmentations...")
    augmenter = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    ])
    print("Generating and saving augmented images...")
    _augment_and_save_img(augmenter, class_images)
    print("Done!")


