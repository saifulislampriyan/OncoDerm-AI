import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained model
#model = load_model('cancer_detection_model.h5')  ## use this if model is saved in .h5
#model = load_model('cancer_detection_model.keras')  ## use this if model is saved in .keras format

# Define target image dimensions (same as used during training)
img_size = (224, 224)

# Streamlit app
st.title("Skin Cancer Detection App")

# Upload image through Streamlit UI
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the uploaded image
    img = Image.open(uploaded_image)
    img = img.resize(img_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array)

    # Predict the class using the loaded model
    prediction = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(prediction)
    predicted_class = predicted_class_index  # Use the index as the predicted class

    # Display the prediction
    class_labels = {
        0: 'actinic keratosis',
        1: 'basal cell carcinoma',
        2: 'dermatofibroma',
        3: 'melanoma',
        4: 'nevus',
        5: 'pigmented benign keratosis',
        6: 'seborrheic keratosis',
        7: 'squamous cell carcinoma',
        8: 'vascular lesion'
    }

    st.write(f"Predicted Class: {class_labels[predicted_class]}")
