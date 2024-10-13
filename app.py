import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image
import cv2

# Define the paths
model_path = r'C:\Users\pakcomp\Downloads\Mango_db\MangoLeafBD Dataset\trained_plant_disease_model.pkl'
label_encoder_path = r'C:\Users\pakcomp\Downloads\Mango_db\MangoLeafBD Dataset\label_encoder.pkl'
image_size = (128, 128)

# Load the model
def load_model():
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

# Load the label encoder
def load_label_encoder():
    with open(label_encoder_path, "rb") as file:
        label_encoder = pickle.load(file)
    return label_encoder

# Function to preprocess and predict image
def predict_image(img_path, model, label_encoder):
    img = Image.open(img_path)
    img = img.resize(image_size)
    img = np.array(img)
    
    if img.ndim == 2:  # Convert grayscale to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    img = img.flatten().reshape(1, -1)
    
    prediction = model.predict(img)
    label = label_encoder.inverse_transform(prediction)
    return label[0]

# Load model and label encoder
model = load_model()
label_encoder = load_label_encoder()

# Streamlit App
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("MANGO LEAF DISEASE RECOGNITION SYSTEM")
    

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    This dataset consists of images of healthy and diseased mango leaves categorized into several classes.
    Our mission is to help in identifying Mango Leaf diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a leave with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Mango Leaf Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "Disease Recognition":
    st.title("Leaf Disease Recognition")
test_image = st.file_uploader("Upload a leaf image:", type=["jpg", "png", "jpeg"])
if test_image:
    st.image(test_image, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        model = load_model()
        label_encoder = load_label_encoder()
        result = predict_image(test_image, model, label_encoder)
        
        st.write(f"Prediction: {result}")
        if result != 'Healthy':
            disease_info = {
                'Powdery Mildew': "Powdery mildew is a fungal disease that affects a wide range of plants, causing white powdery spots on leaves and stems.",
                'Anthracnose': "Anthracnose is a group of fungal diseases that cause dark, sunken lesions on leaves, stems, flowers, and fruits.",
                'Bacterial Canker': "Bacterial canker is a disease caused by bacteria, leading to sunken, oozing lesions on leaves, branches, and fruit.",
                'Cutting Weevil': "Cutting weevil is an insect pest that causes damage by cutting through the stems of young plants, leading to wilting and death.",
                'Die Back': "Dieback is a symptom of various diseases, characterized by the progressive death of shoots, branches, and roots from the tip backward."
            }
            st.write(f"Disease Information: {disease_info[result]}")
        else:
            st.write("The leaf is healthy!")
