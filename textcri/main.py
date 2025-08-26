import os
import streamlit as st
import PyPDF2
import io
from openai import OpenAI
from dotenv import load_dotenv
import time
load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon=":books:",layout="centered")
st.title("PDF Chatbot :books:")
st.markdown("testing text dont mind me")

groq_api_key = os.getenv("GROQ_API_KEY")

uploaded_file=st.file_uploader("Choose a PDF file", type=["pdf","txt"])

job_role =st.text_input("Enter your job role (optional)")
analyze=st.button("Analyze")
def extract_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def analyze_pdf(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

if analyze and uploaded_file is not None:
    try:
        file_content = analyze_pdf(uploaded_file)
        if not file_content.strip():
            st.error("The uploaded PDF file is empty or contains no extractable text.")
            st.stop()
        prompt=f"""analyze this resume and provide feedback
        focus on skills, experience, and education
        then suggest specific improvements for {job_role if job_role else "a general job role"}
        resume content:
        {file_content}

        """

        client = OpenAI( api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system","content": "pretend you are rick sanchez but instead you encourage the user to think about getting a JOB"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
        )
        st.markdown("### Analysis Result:")
        st.markdown(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error occurred: {e}")