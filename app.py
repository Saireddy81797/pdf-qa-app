import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="📄 Ask Questions from PDF", layout="wide")
st.title("📄 Ask Questions from PDF (Offline, No API)")

@st.cache_resource
def load_model():
    pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_length=1024)
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_text(raw_text)

    with st.spinner("Indexing document..."):
        embeddings = load_embeddings()
        docsearch = FAISS.from_texts(texts, embeddings)

    query = st.text_input("Ask a question from the PDF:")
    if query:
        with st.spinner("Searching and generating answer..."):
            docs = docsearch.similarity_search(query, k=4)
            llm = load_model()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=query)
            st.markdown("### 📢 Answer:")
            st.write(answer)
