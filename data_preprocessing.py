import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import shutil
import os
import argparse


# Create output directories
def create_class_dirs(base_dir, classes):
    for cls in classes:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)


def clean_dataset(df):
    # Remove rows with na values in 'dx', 'image_id', or 'lesion_id'
    df.dropna(subset=['dx', 'image_id', 'lesion_id'], inplace=True)
    # Remove rows with invalid 'dx' values
    valid_dx_values = {'nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df'}
    df = df[df['dx'].isin(valid_dx_values)]
    return df


def split_dataset(metadata_file, image_dir, output_dir, train_ratio=0.7, val_ratio=0.2, random_state=42):
    # Calculate the test ratio
    test_ratio = 1 - train_ratio - val_ratio

    # Load the dataset metadata
    df = pd.read_csv(metadata_file)

    # Clean the dataset
    df = clean_dataset(df)

    # Check ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Train, validation, and test ratios must sum to 1."

    # Initialize stratified splitter
    skf = StratifiedGroupKFold(n_splits=int(1 / (1 - train_ratio)), shuffle=True, random_state=random_state)

    # Create stratified split
    for train_index, temp_index in skf.split(df, df['dx'], groups=df['lesion_id']):
        train_df = df.iloc[train_index]
        temp_df = df.iloc[temp_index]
        break

    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    skf_temp = StratifiedGroupKFold(n_splits=int(1 / (1 - val_ratio_adjusted)), shuffle=True, random_state=random_state)

    for val_index, test_index in skf_temp.split(temp_df, temp_df['dx'], groups=temp_df['lesion_id']):
        val_df = temp_df.iloc[val_index]
        test_df = temp_df.iloc[test_index]
        break

    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    classes = df['dx'].unique()

    create_class_dirs(train_dir, classes)
    create_class_dirs(val_dir, classes)
    create_class_dirs(test_dir, classes)

    # Copy images to respective directories
    def copy_images(df, target_dir):
        for _, row in df.iterrows():
            image_path = os.path.join(image_dir, row['image_id'] + '.jpg')
            class_dir = os.path.join(target_dir, row['dx'])
            shutil.copy(image_path, class_dir)

    copy_images(train_df, train_dir)
    copy_images(val_df, val_dir)
    copy_images(test_df, test_dir)

    # Save the splits metadata
    train_df.to_csv(os.path.join(output_dir, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_metadata.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_metadata.csv'), index=False)

    print(
        f"Dataset split completed. Train: {len(train_df)} images, Val: {len(val_df)} images, Test: {len(test_df)} images.")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split the HAM10000 dataset for training YOLO.')
    parser.add_argument('--metadata_file', type=str, required=True, help='Path to the metadata CSV file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the split datasets.')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Proportion of data to be used for training.')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Proportion of data to be used for validation.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility.')

    args = parser.parse_args()

    split_dataset(
        metadata_file=args.metadata_file,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_state
    )
