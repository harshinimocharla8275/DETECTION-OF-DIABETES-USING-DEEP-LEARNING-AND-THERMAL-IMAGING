import os
import cv2
import numpy as np

# Paths to datasets
DIABETIC_PATH = 'C:/Users/harsh/FYP/data/diabetic dataset'
NON_DIABETIC_PATH = 'C:/Users/harsh/FYP/data/nondiabetic dataset'
PROCESSED_PATH = 'C:/Users/harsh/FYP/data/processed'

# Create processed folder if it doesn't exist
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Function to preprocess and save original images
def preprocess_and_save(dataset_path, label):
    """Preprocess images (resize) and save them to the processed directory without augmentation."""
    # Create the label directory (diabetic or nondiabetic)
    label_path = os.path.join(PROCESSED_PATH, label)
    os.makedirs(label_path, exist_ok=True)

    for individual in os.listdir(dataset_path):
        individual_path = os.path.join(dataset_path, individual)
        if os.path.isdir(individual_path):
            for img_file in os.listdir(individual_path):
                img_path = os.path.join(individual_path, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"Warning: Failed to load image: {img_path}")
                    continue
                
                # Resize the image to 224x224 (EfficientNetB0 input size)
                image = cv2.resize(image, (224, 224))
                
                # Ensure the filename has a proper extension
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_file = img_file + '.png'
                
                # Save the original image to the processed directory
                save_path = os.path.join(label_path, img_file)
                cv2.imwrite(save_path, image)
                print(f"Saved original image: {save_path}")
# Process both diabetic and non-diabetic datasets
preprocess_and_save(DIABETIC_PATH, 'diabetic')
preprocess_and_save(NON_DIABETIC_PATH, 'nondiabetic')
print("Preprocessing complete. Original images saved.")