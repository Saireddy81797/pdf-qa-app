import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="📄 PDF Chatbot", layout="wide")
st.title("📄 Ask Questions from PDF (No API)")

# Step 1: Upload PDF
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf:
    # Step 2: Extract text from PDF
    reader = PdfReader(pdf)
    raw_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    # Step 3: Split text
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    # Step 4: Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunks, embedding=embeddings)

    # Step 5: Load HuggingFace QA model
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base", max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    # Step 6: Input box for questions
    question = st.text_input("Ask a question from the PDF:")
    if question:
        result = qa.run(question)
        st.markdown("### ✅ Answer:")
        st.write(result)
