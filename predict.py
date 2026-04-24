import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

with open("class_names.json", "r") as f:
    class_indices = json.load(f)

class_names = sorted(class_indices, key=class_indices.get)

# Load model
@st.cache_resource
def load_cattle_model():
    return load_model("cattle_buffalo_model.h5")

model = load_cattle_model()

st.title("Cattle & Buffalo Breed Detector")

uploaded_file = st.file_uploader("Upload an image of cattle/buffalo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Show image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = image.load_img(uploaded_file, target_size=(224,224))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # Prediction
        pred = model.predict(x)
        pred_prob = np.max(pred)
        pred_class = np.argmax(pred)

        THRESHOLD = 0.7

        if pred_prob < THRESHOLD:
            st.error("❌ This does not appear to be a valid cattle/buffalo image.")
        else:
            breed = class_names[pred_class]

            st.success(f"✅ Predicted Breed: **{breed}**")
            st.info(f"Confidence: {pred_prob*100:.2f}%")

            st.subheader("Prediction Probabilities:")
            for i, prob in enumerate(pred[0]):
                st.write(f"{class_names[i]}: {prob*100:.2f}%")

    except Exception as e:
        st.error(f"Error processing image: {e}")