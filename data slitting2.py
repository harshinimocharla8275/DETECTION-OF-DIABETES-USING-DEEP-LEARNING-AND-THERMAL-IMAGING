import os
import shutil
import random

# Source and destination directories
SOURCE_DIR = 'C:/Users/harsh/FYP/data/processed'
DEST_DIR = 'C:/Users/harsh/FYP/data/split data'

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Valid image extensions
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

def split_data():
    """Split the dataset into train, val, and test sets."""
    # Create destination directories
    for split in ['train', 'val', 'test']:
        for class_name in ['diabetic', 'nondiabetic']:
            os.makedirs(os.path.join(DEST_DIR, split, class_name), exist_ok=True)

    # Process each class
    for class_name in ['diabetic', 'nondiabetic']:
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue

        # Get all images
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS)]
        if not images:
            print(f"No valid images found in {class_dir}")
            continue

        # Shuffle the images
        random.shuffle(images)

        # Calculate split sizes
        total_images = len(images)
        train_size = int(total_images * TRAIN_RATIO)
        val_size = int(total_images * VAL_RATIO)
        test_size = total_images - train_size - val_size

        # Split the images
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        # Copy images to respective directories
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(DEST_DIR, 'train', class_name, img))
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(DEST_DIR, 'val', class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(DEST_DIR, 'test', class_name, img))

        print(f"Class {class_name}: {total_images} images")
        print(f"  Train: {len(train_images)}")
        print(f"  Val: {len(val_images)}")
        print(f"  Test: {len(test_images)}")

if __name__ == "__main__":
    # Clear the destination directory to avoid duplicates
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
        print(f"Cleared existing directory: {DEST_DIR}")
    
    split_data()
    print("Data splitting complete.")