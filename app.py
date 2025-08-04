import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Load embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load QA chain
@st.cache_resource
def load_qa():
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base", max_length=512)
    local_llm = HuggingFacePipeline(pipeline=qa_pipeline)
    return load_qa_chain(llm=local_llm, chain_type="stuff")

# Streamlit app UI
st.title("📄 Offline PDF Question Answering App (No API Required)")

uploaded_file = st.file_uploader("📤 Upload your PDF file", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(raw_text)

    embeddings = load_embeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    st.success("✅ PDF uploaded and processed. Ask your question below!")

    query = st.text_input("❓ Enter your question:")
    if query:
        docs = docsearch.similarity_search(query, k=3)
        qa_chain = load_qa()
        response = qa_chain.run(input_documents=docs, question=query)
        st.write("### 📌 Answer:")
        st.write(response)
