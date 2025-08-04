import streamlit as st
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="📄 PDF Chatbot", layout="wide")
st.title("📄 Ask Questions from PDF (Offline, No API)")

# Step 1: Upload PDF
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf:
    # Step 2: Extract text from PDF
    reader = PdfReader(pdf)
    raw_text = ""
    for page in reader.pages:
        if page.extract_text():
            raw_text += page.extract_text()

    # Step 3: Split text
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    # Step 4: Convert to Documents
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Step 5: Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embedding=embeddings)

    # Step 6: Load lightweight QA model
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base", max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    # Step 7: Ask questions
    question = st.text_input("Ask a question from the PDF:")
    if question:
        result = qa.run(question)
        st.markdown("### ✅ Answer:")
        st.write(result)
