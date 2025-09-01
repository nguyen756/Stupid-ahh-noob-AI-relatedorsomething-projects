import streamlit as st
import easyocr
import fitz
from PIL import Image
import io
import numpy as np
import cv2

st.title("📄 OCR with EasyOCR")
st.write("Upload an **image** or **PDF** to extract text.")

# Resize function for large images
def resize_image(image, max_dim=1600):
    width, height = image.size
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.LANCZOS)
    return image

# Cache EasyOCR model to avoid reloading each time
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en', 'vi'])

reader = load_reader()

uploaded_file = st.file_uploader("Upload image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        st.info("Processing PDF...")

        pdf_data = uploaded_file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        extracted_text = ""

        for i, page in enumerate(doc, start=1):
            st.subheader(f"Page {i}")

            # Extract selectable text
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

                        # Convert image bytes to PIL Image and resize
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        image = resize_image(image)
                        st.image(image, caption=f"Page {i} - Image {j}", use_container_width=True)

                        # OCR with EasyOCR
                        np_image = np.array(image)
                        results = reader.readtext(np_image)
                        ocr_text = "\n".join([res[1] for res in results])

                        st.text_area(f"OCR Text (Page {i} - Image {j})", ocr_text, height=150)
                        extracted_text += f"\n\n--- Page {i}, Image {j} ---\n{ocr_text}"
                else:
                    st.warning(f"No images found on page {i}.")

        # Download button for all extracted text
        st.download_button(
            "Download Extracted Text",
            data=extracted_text,
            file_name="hybrid_output.txt",
            mime="text/plain"
        )

    else:
        # IMAGE BRANCH: Only runs if uploaded file is not PDF
        st.info("Processing Image...")

        image = Image.open(uploaded_file).convert("RGB")
        image = resize_image(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        np_image = np.array(image)
        results = reader.readtext(np_image)
        extracted_text = "\n".join([res[1] for res in results])

        st.subheader("Extracted Text")
        st.text_area("Detected text", extracted_text, height=200)

        st.download_button(
            "Download Extracted Text",
            data=extracted_text,
            file_name="easyocr_output.txt",
            mime="text/plain"
        )
