import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
import sys
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from PIL import Image
import logging
import matplotlib.pyplot as plt
import io
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from scipy.stats import skew
import tensorflow as tf
from fpdf import FPDF
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Python version: {sys.version}")
logger.info(f"TensorFlow version: {tf.__version__}")
# App configuration
UPLOAD_FOLDER = '"C:/Users/harsh/FYP/src\streamlit_app\uploads"'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"Upload folder created: {UPLOAD_FOLDER}")
# Paths
MODEL_PATH = 'C:/Users/harsh/FYP/models/cnn_model/model.keras'
IMG_HEIGHT, IMG_WIDTH = 224, 224
# Load the model
try:
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise
# Create a model for feature extraction
logger.info("Creating feature extraction model...")
layer_name = 'block_3_expand'
feature_extraction_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
logger.info("Feature extraction model created")
# Define normal ranges for thermal image features
NORMAL_RANGES = {
    'contrast': (0, 50),
    'energy': (0.1, 0.5),
    'homogeneity': (0.5, 1.0),
    'correlation': (0.7, 1.0),
    'mean_temp': (30, 34),
    'std_dev': (1, 3),
    'skewness': (-0.5, 0.5),
    'min_temp': (28, 32),
    'max_temp': (32, 36),
    'heel_temp': (30, 34),
    'toe_temp': (28, 32),
    'arch_temp': (30, 34),
    'hotspots': (0, 5),
    'circulation_score': (0.6, 1.0)
}
def recommend_exercises(circulation_score, max_temp):
    """Recommend exercises based on circulation score and maximum temperature."""
    recommendations = []

    if max_temp > 37.0:
        recommendations.append({
            'exercise': 'Rest and Elevation',
            'description': 'Rest with feet elevated for 15-20 minutes to reduce inflammation.'
        })
        recommendations.append({
            'exercise': 'Cooling Therapy',
            'description': 'Apply a cool compress to the feet for 10 minutes to lower temperature.'
        })
        recommendations.append({
            'exercise': 'Avoid Weight-Bearing Exercises',
            'description': 'Refrain from walking or standing exercises until the temperature normalizes.'
        })
        return recommendations

    if circulation_score < 0.3 or max_temp < 30.0:
        recommendations.append({
            'exercise': 'Toe Wiggling',
            'description': 'Gently wiggle toes for 5 minutes to stimulate blood flow.'
        })
        recommendations.append({
            'exercise': 'Ankle Rotations',
            'description': 'Rotate ankles clockwise and counterclockwise for 5 minutes to improve circulation.'
        })
        recommendations.append({
            'exercise': 'Seated Leg Raises',
            'description': 'While seated, lift one leg at a time and hold for 10 seconds, repeat 10 times per leg.'
        })
    elif 0.3 <= circulation_score < 0.6:
        recommendations.append({
            'exercise': 'Calf Raises',
            'description': 'Stand and rise onto your toes, hold for 3 seconds, then lower. Repeat 15 times.'
        })
        recommendations.append({
            'exercise': 'Walking',
            'description': 'Take a 10-15 minute walk at a moderate pace to improve circulation.'
        })
        recommendations.append({
            'exercise': 'Foot Massage',
            'description': 'Gently massage the feet for 5-10 minutes to stimulate blood flow.'
        })
    else:
        recommendations.append({
            'exercise': 'Brisk Walking',
            'description': 'Walk briskly for 20-30 minutes to maintain circulation.'
        })
        recommendations.append({
            'exercise': 'Toe Taps',
            'description': 'Tap toes rapidly on the ground for 1 minute, rest, and repeat 5 times.'
        })
        recommendations.append({
            'exercise': 'Stretching',
            'description': 'Perform foot and calf stretches (e.g., towel stretch) for 10 minutes.'
        })

    return recommendations
def extract_features(image_path):
    """Extract features from the thermal image."""
    try:
        img = load_img(image_path, color_mode='grayscale')
        img_array = img_to_array(img).squeeze()

        # Scale pixel values to a realistic temperature range (e.g., 20°C to 40°C)
        temperatures = 20 + (img_array / 255.0) * (40 - 20)

        mean_temp = np.mean(temperatures)
        std_dev = np.std(temperatures)
        skewness = skew(temperatures.flatten())
        min_temp = np.min(temperatures)
        max_temp = np.max(temperatures)

        img_ubyte = img_as_ubyte(temperatures / temperatures.max())
        glcm = graycomatrix(img_ubyte, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        height, width = temperatures.shape
        heel_region = temperatures[height//2:, :width//3]
        toe_region = temperatures[:height//3, width//3:2*width//3]
        arch_region = temperatures[height//3:2*height//3, 2*width//3:]

        heel_temp = np.mean(heel_region)
        toe_temp = np.mean(toe_region)
        arch_temp = np.mean(arch_region)

        threshold = np.percentile(temperatures, 90)
        hotspots = np.sum(temperatures > threshold)

        circulation_score = 1 - (std_dev / max_temp) - (hotspots / temperatures.size)

        return {
            'contrast': contrast,
            'energy': energy,
            'homogeneity': homogeneity,
            'correlation': correlation,
            'mean_temp': mean_temp,
            'std_dev': std_dev,
            'skewness': skewness,
            'min_temp': min_temp,
            'max_temp': max_temp,
            'heel_temp': heel_temp,
            'toe_temp': toe_temp,
            'arch_temp': arch_temp,
            'hotspots': hotspots,
            'circulation_score': circulation_score
        }

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise
def predict_cnn(image_path):
    """Predict whether an image is diabetic or nondiabetic using the trained CNN model."""
    try:
        img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = 'Diabetic' if prediction[0] > 0.5 else 'Non-Diabetic'
        confidence = prediction[0].item() if predicted_class == 'Diabetic' else (1 - prediction[0]).item()

        features = feature_extraction_model.predict(img_array)
        return predicted_class, confidence, features

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
def plot_feature_maps(features, num_filters=6):
    """Plot a subset of feature maps from the intermediate layer."""
    fig, axes = plt.subplots(1, num_filters, figsize=(15, 3))
    for i in range(num_filters):
        if i < features.shape[-1]:
            axes[i].imshow(features[0, :, :, i], cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Filter {i+1}')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf
def generate_pdf_report(foot, data, patient_name, patient_age):
    """Generate a PDF report for the given foot's analysis results."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt=f"{foot.capitalize()} Foot Analysis Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="=" * 40, ln=True, align="C")
    pdf.ln(5)

    # Patient Details
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Patient Details", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Name: {patient_name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {patient_age}", ln=True)
    pdf.ln(5)

    # Prediction and Confidence
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Prediction Results", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction: {data['prediction']}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {data['confidence']:.2%}", ln=True)
    pdf.ln(5)

    # Thermal Image Features
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Thermal Image Features", ln=True)
    pdf.set_font("Arial", size=12)
    features = data['features']
    for feature_name, value in features.items():
        normal_range = NORMAL_RANGES.get(feature_name, None)
        if normal_range:
            min_val, max_val = normal_range
            status = "Normal" if min_val <= value <= max_val else "Abnormal"
            # Replace en dash with hyphen to avoid encoding issues
            normal_range_str = f"Normal Range: {min_val}-{max_val}"
            pdf.cell(
                200, 10,
                txt=f"- {feature_name.replace('_', ' ').title()}: {value:.2f} "
                    f"({status}, {normal_range_str})",
                ln=True
            )
    pdf.ln(5)

    # Exercise Recommendations
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Exercise Recommendations", ln=True)
    pdf.set_font("Arial", size=12)
    for exercise in data['exercises']:
        pdf.cell(200, 10, txt=f"- {exercise['exercise']}: {exercise['description']}", ln=True)
    pdf.ln(5)

    pdf.cell(200, 10, txt="=" * 40, ln=True, align="C")

    # Output to bytes
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output

# Streamlit app
def main():
    st.title("Diabetic Foot Detection")
    st.write("Upload thermal images of your left and right foot to predict diabetic foot conditions.")

    # Patient details input
    st.subheader("Patient Details")
    patient_name = st.text_input("Patient Name", value="John Doe")
    patient_age = st.text_input("Patient Age", value="30")
    st.write("")  # Add some spacing

    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        left_file = st.file_uploader("Left Foot Image", type=["png", "jpg", "jpeg"], key="left")
        if left_file:
            st.image(left_file, caption="Left Foot Preview", use_container_width=True)

    with col2:
        right_file = st.file_uploader("Right Foot Image", type=["png", "jpg", "jpeg"], key="right")
        if right_file:
            st.image(right_file, caption="Right Foot Preview", use_container_width=True)

    # Predict button
    if st.button("Predict"):
        if not left_file and not right_file:
            st.error("Please upload at least one image.")
            return

        if not patient_name or not patient_age:
            st.error("Please provide the patient's name and age.")
            return

        try:
            patient_age = int(patient_age)  # Ensure age is a number
            if patient_age <= 0:
                st.error("Patient age must be a positive number.")
                return
        except ValueError:
            st.error("Patient age must be a valid number.")
            return

        results = {}

        # Process left foot
        if left_file:
            left_filename = os.path.join(UPLOAD_FOLDER, f"left_{left_file.name}")
            with open(left_filename, "wb") as f:
                f.write(left_file.getbuffer())
            logger.info(f"Left foot file uploaded: {left_filename}")

            try:
                predicted_class, confidence, features = predict_cnn(left_filename)
                feature_map_img = plot_feature_maps(features)
                extracted_features = extract_features(left_filename)
                exercises = recommend_exercises(extracted_features['circulation_score'], extracted_features['max_temp'])
                results['left'] = {
                    'image_path': left_filename,
                    'prediction': predicted_class,
                    'confidence': confidence,
                    'feature_map': feature_map_img,
                    'features': extracted_features,
                    'exercises': exercises
                }
            except Exception as e:
                logger.error(f"Prediction failed for left foot: {e}")
                st.error(f"Error processing left foot image: {str(e)}")
            finally:
                if os.path.exists(left_filename):
                    os.remove(left_filename)
                    logger.info(f"Left foot file removed: {left_filename}")

        # Process right foot
        if right_file:
            right_filename = os.path.join(UPLOAD_FOLDER, f"right_{right_file.name}")
            with open(right_filename, "wb") as f:
                f.write(right_file.getbuffer())
            logger.info(f"Right foot file uploaded: {right_filename}")

            try:
                predicted_class, confidence, features = predict_cnn(right_filename)
                feature_map_img = plot_feature_maps(features)
                extracted_features = extract_features(right_filename)
                exercises = recommend_exercises(extracted_features['circulation_score'], extracted_features['max_temp'])
                results['right'] = {
                    'image_path': right_filename,
                    'prediction': predicted_class,
                    'confidence': confidence,
                    'feature_map': feature_map_img,
                    'features': extracted_features,
                    'exercises': exercises
                }
            except Exception as e:
                logger.error(f"Prediction failed for right foot: {e}")
                st.error(f"Error processing right foot image: {str(e)}")
            finally:
                if os.path.exists(right_filename):
                    os.remove(right_filename)
                    logger.info(f"Right foot file removed: {right_filename}")

        # Display results

        if results:
            for foot, data in results.items():
                st.subheader(f"{foot.capitalize()} Foot Analysis")
                
                # Display prediction
                st.write(f"**Prediction**: {data['prediction']}")
                st.write(f"**Confidence**: {data['confidence']:.2%}")

                # Display feature maps
                st.write("**Feature Maps**:")
                st.image(data['feature_map'], use_container_width=True)

                # Display thermal features with normal ranges
                st.write("**Thermal Image Features**:")
                features = data['features']
                for feature_name, value in features.items():
                    normal_range = NORMAL_RANGES.get(feature_name, None)
                    if normal_range:
                        min_val, max_val = normal_range
                        # Check if the value is within the normal range
                        if value < min_val or value > max_val:
                            color = "red"  # Highlight abnormal values in red
                            status = " (Abnormal)"
                        else:
                            color = "green"  # Highlight normal values in green
                            status = " (Normal)"
                        # Replace en dash with hyphen in the display as well
                        normal_range_str = f"Normal Range: {min_val}-{max_val}"
                        st.markdown(
                            f"- **{feature_name.replace('_', ' ').title()}**: "
                            f"<span style='color:{color}'>{value:.2f}</span>"
                            f"{status} ({normal_range_str})",
                            unsafe_allow_html=True
                        )

                # Display exercise recommendations
                st.write("**Exercise Recommendations**:")
                for exercise in data['exercises']:
                    st.write(f"- **{exercise['exercise']}**: {exercise['description']}")

                # Generate and provide a download button for the PDF report
                report = generate_pdf_report(foot, data, patient_name, patient_age)
                st.download_button(
                    label=f"Download {foot.capitalize()} Foot Report (PDF)",
                    data=report,
                    file_name=f"{foot}_foot_analysis_report.pdf",
                    mime="application/pdf"
                )

if __name__ == '__main__':
    logger.info("Starting Streamlit app...")
    main()