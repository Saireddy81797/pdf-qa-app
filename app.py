import streamlit as st
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

st.title("📄 Ask Questions from PDF (Fast, Offline)")

@st.cache_data
def load_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@st.cache_resource
def get_embeddings_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_answer(question, chunks, model):
    chunk_embeddings = model.encode(chunks)
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    best_idx = np.argmax(similarities)
    return chunks[best_idx][:1000]  # return top relevant chunk (max 1000 chars)

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    text = load_pdf_text(uploaded_file)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    model = get_embeddings_model()
    question = st.text_input("Ask a question from the PDF:")
    
    if question:
        with st.spinner("Searching for answer..."):
            answer = get_answer(question, chunks, model)
        st.markdown("**Answer:**")
        st.write(answer)
