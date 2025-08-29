import streamlit as st
import easyocr
import fitz
from PIL import Image
import io
import numpy as np
import cv2

st.title("📄 OCR with EasyOCR")
st.write("Upload an **image** or **PDF** to extract text.")

reader = easyocr.Reader(['en','vi'])  # You can add more like ['en', 'vi']

uploaded_file = st.file_uploader("Upload image or PDF", type=["jpg", "jpeg", "png", "pdf"])


if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        st.info("Processing PDF...")

        pdf_data = uploaded_file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        extracted_text = ""

        for i, page in enumerate(doc, start=1):
            st.subheader(f"Page {i}")

            # Extract selectable text first
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

                        # Convert image bytes to PIL Image
                        image = Image.open(io.BytesIO(image_bytes))
                        st.image(image, caption=f"Page {i} - Image {j}", use_container_width=True)

                        # OCR with EasyOCR
                        results = reader.readtext(np.array(image))
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
        st.info("Processing Image...")

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        results = reader.readtext(np.array(image))
        extracted_text = "\n".join([res[1] for res in results])

        st.subheader("Extracted Text")
        st.text_area("Detected text", extracted_text, height=200)

        st.download_button(
            "Download Extracted Text",
            data=extracted_text,
            file_name="easyocr_output.txt",
            mime="text/plain"
        )
