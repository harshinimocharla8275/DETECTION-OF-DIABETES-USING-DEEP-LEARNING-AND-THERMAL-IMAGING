import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = 'C:/Users/harsh/FYP/models/cnn_model/model.keras'
IMG_HEIGHT, IMG_WIDTH = 224, 224

def predict_cnn(image_path):
    """Predict whether an image is diabetic or nondiabetic using the trained CNN model."""
    try:
        # Check if the image path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load the model
        model = load_model(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")

        # Load and preprocess the image
        img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Rescale to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = 'Diabetic' if prediction[0] > 0.5 else 'Non-Diabetic'
        # Extract scalar value using .item() to avoid deprecation warning
        confidence = prediction[0].item() if predicted_class == 'Diabetic' else (1 - prediction[0]).item()

        logger.info(f"Prediction for {image_path}: {predicted_class} (confidence: {confidence:.2f})")
        return predicted_class, confidence

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

if __name__ == "__main__":
    # Test the model on a sample image
    sample_image_path = 'C:/Users/harsh/FYP/data/split data/test/diabetic/DM019_F_R.png'  # Choose a test image
    try:
        predicted_class, confidence = predict_cnn(sample_image_path)
        print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")