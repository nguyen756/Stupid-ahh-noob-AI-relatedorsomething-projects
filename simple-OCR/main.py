import streamlit as st
import easyocr
import fitz
from PIL import Image
import io
import numpy as np
from st_img_pastebutton import paste
import base64
import os
import streamlit as st
import io
from openai import OpenAI
from dotenv import load_dotenv




load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets["GROQ_API_KEY"]


st.set_page_config(page_title="OCR with EasyOCR", layout="wide")
st.title("📄 OCR with EasyOCR")
st.write("Upload an **image/PDF** or paste an image from your clipboard to extract text.")

# Resize large images
def resize_image(image, max_dim=1600):
    width, height = image.size
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.LANCZOS)
    return image

# Cache EasyOCR model
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en', 'vi','ja'])

reader = load_reader()

# Sidebar: choose input method
method = st.sidebar.radio("Select input method:", ["Upload file", "Paste image"])

# ===== UPLOAD FILE =====
if method == "Upload file":
    uploaded_file = st.file_uploader("Upload image or PDF", type=["jpg", "jpeg", "png", "pdf"])
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            st.info("Processing PDF...")
            pdf_data = uploaded_file.read()
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            extracted_text = ""

            for i, page in enumerate(doc, start=1):
                st.subheader(f"Page {i}")
                page_text = page.get_text().strip()

                if page_text:
                    st.text_area(f"Text (Page {i})", page_text, height=150)
                    extracted_text += f"\n\n--- Page {i} (Text) ---\n{page_text}"
                else:
                    st.warning(f"No direct text found on page {i}. Running OCR...")
                    images = page.get_images(full=True)
                    if images:
                        for j, img in enumerate(images, start=1):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            image = resize_image(image)
                            st.image(image, caption=f"Page {i} - Image {j}", use_container_width=True)
                            np_image = np.array(image)
                            results = reader.readtext(np_image)
                            ocr_text = "\n".join([res[1] for res in results])
                            st.text_area(f"OCR Text (Page {i} - Image {j})", ocr_text, height=150)
                            extracted_text += f"\n\n--- Page {i}, Image {j} ---\n{ocr_text}"
                    else:
                        st.warning(f"No images found on page {i}.")

            st.download_button(
                "Download Extracted Text",
                data=extracted_text,
                file_name="hybrid_output.txt",
                mime="text/plain"
            )

        else:
            st.info("Processing Image...")
            image = Image.open(uploaded_file).convert("RGB")
            image = resize_image(image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Extract Text"):
                np_image = np.array(image)
                results = reader.readtext(np_image)
                extracted_text = "\n".join([res[1] for res in results])
                st.text_area("Detected text", extracted_text, height=200)
                st.download_button(
                    "Download Extracted Text",
                    data=extracted_text,
                    file_name="easyocr_output.txt",
                    mime="text/plain"
                )

elif method == "Paste image":
    image_data = paste(label="📋 Click here, then paste your image (Ctrl+V)", key="pastebox")

    if image_data:
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(binary_data)).convert("RGB")
        image = resize_image(image)
        st.image(image, caption="Pasted Image", use_container_width=True)

        if st.button("Extract Text"):
            np_image = np.array(image)
            results = reader.readtext(np_image)
            extracted_text = "\n".join([res[1] for res in results])
            st.text_area("Detected Text", extracted_text, height=200)
            st.download_button(
                "Download Extracted Text",
                data=extracted_text,
                file_name="pasted_image_ocr.txt",
                mime="text/plain"
            )
            prompt=f""" if any text in this image is a question, answer it, if not, just say "no question found", if there are information, try to elbaorate
            image content:
            {extracted_text}
            """

            client = OpenAI(
            api_key=gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            response = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[
                {"role": "system","content": "pretend you are rick sanchez, but you dont say Wubba Lubba Dub Dub, instead you will try to be the biggest asshole but still actively trying to answer me"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            )
            st.markdown("### Analysis Result:")
            st.markdown(response.choices[0].message.content)

    else:
        st.info("Click the box above, copy an image, and press Ctrl+V.")
