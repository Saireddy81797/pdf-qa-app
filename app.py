import streamlit as st
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

st.title("📄 Ask Questions from PDF (Smart Answering)")

@st.cache_data
def load_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@st.cache_resource
def get_model():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return embedder, qa_pipeline

def get_best_chunk(question, chunks, embedder):
    chunk_embeddings = embedder.encode(chunks)
    question_embedding = embedder.encode([question])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    best_idx = np.argmax(similarities)
    return chunks[best_idx]

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    text = load_pdf_text(uploaded_file)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    embedder, qa_pipeline = get_model()

    question = st.text_input("Ask a question from the PDF:")

    if question:
        with st.spinner("Thinking..."):
            best_chunk = get_best_chunk(question, chunks, embedder)
            result = qa_pipeline(question=question, context=best_chunk)
            answer = result['answer']
        st.markdown("**Answer:**")
        st.write(answer)
