import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

# Load QA model
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Streamlit UI
st.set_page_config(page_title="📄 PDF Q&A")
st.title("📄 Ask Questions from PDF (No API Key)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    st.success("✅ PDF Loaded")

    qa_pipeline = load_qa_pipeline()

    query = st.text_input("Ask a question from the PDF:")
    if query:
        with st.spinner("Searching..."):
            result = qa_pipeline(question=query, context=text)
            st.write("**Answer:**", result['answer'])
