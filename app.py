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

# Load question-answering chain
@st.cache_resource
def load_qa():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small",
        max_length=256
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return load_qa_chain(llm=llm, chain_type="stuff")

# Streamlit app layout
st.set_page_config(page_title="📄 Ask Questions from PDF", layout="centered")
st.title("📄 Ask Questions from PDF (Offline, No API)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    # Create vector store
    embeddings = load_embeddings()
    db = FAISS.from_texts(texts, embeddings)

    # Ask questions
    query = st.text_input("Ask a question from the PDF:")
    if query:
        with st.spinner("Searching..."):
            docs = db.similarity_search(query, k=3)
            chain = load_qa()
            answer = chain.run(input_documents=docs, question=query)
            st.success("Answer:")
            st.write(answer)
