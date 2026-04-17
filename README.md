🧠 Detection of Diabetes Using Deep Learning and Thermal Imaging
📌 Project Overview

This project focuses on detecting diabetes using deep learning techniques applied to thermal imaging data. 
The system analyzes thermal images of the human body to identify patterns associated with diabetic and non-diabetic conditions.

🎯 Objectives
To build a deep learning model for diabetes detection
To analyze thermal images for classification
To improve early diagnosis using AI
🧪 Technologies Used
Python
TensorFlow / Keras
Streamlit (for web app)
OpenCV
NumPy, Pandas


📂 Project Structure
FYP/
│── data/                  # Dataset (training, validation)
│── models/                # Trained CNN model
│── src/
│   └── streamlit_app/     # Streamlit application
│       ├── app.py
│       ├── uploads/
│       └── requirements.txt
│── viva questions.txt
│── README.md


🤖 Model Details
Model: Convolutional Neural Network (CNN)
Framework: Keras


Output: Binary classification (Diabetic / Non-Diabetic)
🚀 How to Run the Project
🔹 1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd FYP
🔹 2. Install dependencies
pip install -r src/streamlit_app/requirements.txt
🔹 3. Run the Streamlit app
streamlit run src/streamlit_app/app.py


📊 Dataset
Thermal images categorized into:
Diabetic
Non-Diabetic
📸 Features
Upload thermal image
Predict diabetic condition
User-friendly interface


⚠️ Limitations
Requires good quality thermal images
Model accuracy depends on dataset size
🔮 Future Work
Improve model accuracy
Use larger dataset
Deploy on cloud


👩‍💻 Author
Harshini Mocharla
