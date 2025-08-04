import streamlit as st
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

st.title("📄 Smart PDF Q&A (Full Sentence Answers)")

@st.cache_data
def load_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return embedder, summarizer

def get_top_chunks(question, chunks, embedder, top_k=2):
    chunk_embeddings = embedder.encode(chunks)
    question_embedding = embedder.encode([question])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    text = load_pdf_text(uploaded_file)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    embedder, summarizer = load_models()

    question = st.text_input("Ask a question from the PDF:")

    if question:
        with st.spinner("Thinking..."):
            top_chunks = get_top_chunks(question, chunks, embedder)
            combined_context = " ".join(top_chunks)

            prompt = f"Question: {question}\nContext: {combined_context}\nAnswer:"
            summary = summarizer(prompt, max_length=150, min_length=60, do_sample=False)[0]['summary_text']
        
        st.markdown("**Answer:**")
        st.write(summary)
