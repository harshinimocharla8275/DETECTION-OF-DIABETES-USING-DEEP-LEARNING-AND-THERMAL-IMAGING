import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import logging
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = 'C:/Users/harsh/FYP/data/split data'
MODEL_PATH = 'C:/Users/harsh/FYP/models/cnn_model/model.keras'
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 8
EPOCHS = 75  # Increased to 75 (25 + 25 + 25)
NUM_FOLDS = 5
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

def build_mobilenet_model():
    """Build a MobileNetV2 model with a custom head for binary classification."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model, base_model

def get_all_image_paths_and_labels():
    """Get all image paths and labels from the train and val directories."""
    image_paths = []
    labels = []

    for split in ['train', 'val']:
        for label, class_name in enumerate(['nondiabetic', 'diabetic']):
            class_dir = os.path.join(DATA_PATH, split, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Directory not found: {class_dir}")
                continue
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(VALID_EXTENSIONS):
                    logger.warning(f"Skipping file with invalid extension: {img_name}")
                    continue
                img_path = os.path.abspath(os.path.join(class_dir, img_name))
                img = cv2.imread(img_path)
                if img is None:
                    logger.error(f"Cannot load image: {img_path}")
                    continue
                image_paths.append(img_path)
                labels.append(label)

    return np.array(image_paths), np.array(labels)

def train_cnn():
    """Train the MobileNetV2 model using k-fold cross-validation."""
    try:
        # No data augmentation for training
        train_datagen = ImageDataGenerator(rescale=1./255)

        # Only rescaling for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)

        # Load test data with explicit class indices
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(DATA_PATH, 'test'),
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False,
            classes=['nondiabetic', 'diabetic']
        )

        # Get all image paths and labels from train and val sets
        image_paths, labels = get_all_image_paths_and_labels()
        logger.info(f"Total images for training/validation: {len(image_paths)}")
        if len(image_paths) == 0:
            raise ValueError("No valid images found in train/val directories")

        # Log a few paths for debugging
        logger.info(f"Sample image paths: {image_paths[:3]}")
        logger.info(f"Sample labels: {labels[:3]}")

        # K-Fold Cross-Validation
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths), 1):
            logger.info(f"Training Fold {fold}/{NUM_FOLDS}...")

            # Split data for this fold
            train_paths, val_paths = image_paths[train_idx], image_paths[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            # Create DataFrames with string labels
            train_df = pd.DataFrame({
                'filename': train_paths,
                'class': train_labels.astype(str)
            })
            val_df = pd.DataFrame({
                'filename': val_paths,
                'class': val_labels.astype(str)
            })

            # Log DataFrame info for debugging
            logger.info(f"Fold {fold} Train DataFrame:\n{train_df.head()}")
            logger.info(f"Fold {fold} Val DataFrame:\n{val_df.head()}")

            # Create generators for this fold
            train_generator = train_datagen.flow_from_dataframe(
                dataframe=train_df,
                x_col='filename',
                y_col='class',
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                shuffle=True,
                validate_filenames=True,
                classes=['0', '1']
            )

            val_generator = val_test_datagen.flow_from_dataframe(
                dataframe=val_df,
                x_col='filename',
                y_col='class',
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                shuffle=False,
                validate_filenames=True,
                classes=['0', '1']
            )

            # Verify that the generators have data
            logger.info(f"Fold {fold} Train Generator: {train_generator.n} images")
            logger.info(f"Fold {fold} Val Generator: {val_generator.n} images")
            if train_generator.n == 0 or val_generator.n == 0:
                raise ValueError(f"Generator has no data in Fold {fold}")

            # Compute class weights for this fold with adjustment
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.array([0, 1]),
                y=train_labels
            )
            # Adjust class weights to give more importance to diabetic class
            class_weights = {0: class_weights[0], 1: class_weights[1] * 1.2}  # Increase weight for diabetic class
            logger.info(f"Fold {fold} Class Weights: {class_weights}")

            # Build the model
            model, base_model = build_mobilenet_model()
            if fold == 1:
                model.summary()

            # Callbacks
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )

            # Step 1: Train the head
            logger.info(f"Fold {fold}: Training the head of the model...")
            history = model.fit(
                train_generator,
                epochs=EPOCHS // 3,  # 25 epochs for head training
                validation_data=val_generator,
                class_weight=class_weights,
                callbacks=[lr_scheduler, early_stopping]
            )

            # Step 2: Fine-tune the model (first stage)
            logger.info(f"Fold {fold}: Fine-tuning the later layers (stage 1)...")
            for layer in base_model.layers[-100:]:  # Fine-tune the last 100 layers
                layer.trainable = True
            model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            history_fine1 = model.fit(
                train_generator,
                epochs=EPOCHS // 3,  # 25 epochs for first fine-tuning stage
                validation_data=val_generator,
                class_weight=class_weights,
                callbacks=[lr_scheduler, early_stopping]
            )

            # Step 3: Fine-tune the model (second stage)
            logger.info(f"Fold {fold}: Fine-tuning the later layers (stage 2)...")
            model.compile(optimizer=tf.keras.optimizers.Adam(2e-5),  # Slightly higher learning rate
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            history_fine2 = model.fit(
                train_generator,
                epochs=EPOCHS // 3,  # 25 epochs for second fine-tuning stage
                validation_data=val_generator,
                class_weight=class_weights,
                callbacks=[lr_scheduler, early_stopping]
            )

            # Evaluate on test data
            test_loss, test_accuracy = model.evaluate(test_generator)
            logger.info(f"Fold {fold} Test Accuracy: {test_accuracy:.2f}")
            fold_accuracies.append(test_accuracy)

            # Save the model from the best fold
            if fold == np.argmax(fold_accuracies) + 1:
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                model.save(MODEL_PATH)
                logger.info(f"Best model from Fold {fold} saved to {MODEL_PATH}")

        # Log average test accuracy across folds
        avg_accuracy = np.mean(fold_accuracies)
        logger.info(f"Average Test Accuracy across {NUM_FOLDS} folds: {avg_accuracy:.2f}")

    except Exception as e:
        logger.error(f"CNN training failed: {e}")
        raise

if __name__ == "__main__":
    train_cnn()