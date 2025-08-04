import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="📄 Ask Questions from PDF", layout="centered")

st.title("📄 Ask Questions from PDF (Fast, Accurate)")
st.write("Upload a PDF and ask questions. Powered by OpenAI.")

# Sidebar for OpenAI key
with st.sidebar:
    st.header("🔐 API Configuration")
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    st.markdown(
        "[Get your OpenAI key](https://platform.openai.com/account/api-keys)"
    )

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            raw_text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_texts(texts, embeddings)

    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.3)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    st.success("✅ PDF processed successfully!")

    query = st.text_input("📌 Ask a question from the PDF:")
    if query:
        with st.spinner("Thinking..."):
            result = qa.run(query)
            st.markdown("### 💬 Answer")
            st.write(result)
