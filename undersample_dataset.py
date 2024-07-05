import argparse
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import random


def _extract_img_labels(metadata_df, img_dir):
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


# Undersample Overrepresented Classes by Deleting Files
def _undersample_classes(class_images, metadata_df, max_images_per_class):
    rows_to_delete = []
    for label, images in class_images.items():
        if len(images) > max_images_per_class:
            # Randomly sample images to keep
            images_to_keep = random.sample(images, max_images_per_class)
            # Determine images to delete
            images_to_delete = set(images) - set(images_to_keep)
            for img_path in images_to_delete:
                os.remove(img_path)
                # Find the corresponding row in the DataFrame and mark for deletion
                image_id = os.path.basename(img_path).replace('.jpg', '')
                row_index = metadata_df[metadata_df['image_id'] == image_id].index
                rows_to_delete.extend(row_index)

    # Delete the rows from the DataFrame
    metadata_df.drop(rows_to_delete, inplace=True)
    return metadata_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Undersampling the train split of HAM10000 dataset for a more balanced dataset.')
    parser.add_argument('--metadata_file', type=str, required=True, help='Path to the metadata CSV file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the images.')
    parser.add_argument('--target_size', type=int, required=True,
                        help='Minimum number of samples resulting from the undersampling')
    args = parser.parse_args()
    print("Loading train split info...")
    metadata_df = pd.read_csv(args.metadata_file)
    train_img, train_labels = _extract_img_labels(metadata_df, args.image_dir)
    # Organize images by class
    class_images = defaultdict(list)
    for img_path, label in zip(train_img, train_labels):
        class_images[label].append(img_path)
    # Set a seed for reproducibility
    random.seed(42)
    # Undersample classes and update DataFrame
    updated_metadata_df = _undersample_classes(class_images, metadata_df, args.target_size)
    # Save the updated DataFrame to CSV
    updated_metadata_df.to_csv(args.metadata_file, index=False)
    print(f"Updated metadata saved to {args.metadata_file}")

