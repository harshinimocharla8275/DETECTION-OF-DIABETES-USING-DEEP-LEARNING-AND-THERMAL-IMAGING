import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
# Paths
DATA_PATH = 'C:/Users/harsh/FYP/data/split data'
MODEL_PATH = 'C:/Users/harsh/FYP/models/cnn_model/model.keras'
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 8
def generate_classification_report():
    """Generate a classification report for the test set."""
    # Load the model
    model = load_model(MODEL_PATH)

    # Data generator for test set
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_PATH, 'test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        classes=['nondiabetic', 'diabetic']
    )

    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    # Get filenames
    filenames = test_generator.filenames

    # Create a DataFrame with predictions
    results_df = pd.DataFrame({
        'Filename': filenames,
        'True Label': y_true,
        'Predicted Label': y_pred,
        'Correct': y_true == y_pred
    })
    print("\nPrediction Results:")
    print(results_df)

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['nondiabetic', 'diabetic']))

    # Generate confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    generate_classification_report()