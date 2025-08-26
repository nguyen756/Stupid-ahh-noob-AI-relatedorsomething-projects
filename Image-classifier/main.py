import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import ( MobileNetV2, preprocess_input, decode_predictions)
from PIL import Image


def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    image= np.expand_dims(image, axis=0)
    return image
def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions=model.predict(processed_image) 
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions 
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None
def main():
    st.set_page_config(page_title="Image Classifier", page_icon="🦈", layout="centered") 
    st.title("Image Classifier 🦈")
    st.write("upload your image")
    @st.cache_resource
    def get_model():
        model = load_model()
        return model
    
    model = get_model()
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file is not None:
        image = st.image(file,caption="your image",use_container_width=True)
        button = st.button("classify image")
        if button:
            with st.spinner("classifying..."):
                image = Image.open(file)
                predictions = classify_image(model,image)
                if predictions:
                    st.subheader("Predictions:")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: **{score*100:.2f}%**")
if __name__ == "__main__":
    main()
