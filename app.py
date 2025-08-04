import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="📄 PDF Q&A App", layout="centered")
st.title("📄 Ask Questions from PDF (Fast, Accurate)")

with st.sidebar:
    openai_api_key = st.text_input("🔐 Enter OpenAI API Key", type="password")
    st.markdown("[Get API key](https://platform.openai.com/account/api-keys)")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

uploaded_file = st.file_uploader("📤 Upload your PDF file", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(text)

    question = st.text_input("💬 Ask a question:")
    if question:
        with st.spinner("Answering..."):
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=[{"page_content": chunk} for chunk in docs], question=question)
            st.write("📘 **Answer:**")
            st.success(response)
