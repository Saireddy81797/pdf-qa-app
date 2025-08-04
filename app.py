import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Title
st.set_page_config(page_title=" Ask Questions from PDF", layout="centered")
st.title(" Ask Questions from PDF (Offline, No API)")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        if page.extract_text():
            raw_text += page.extract_text()

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_text(raw_text)

    # Load embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)

    # Load QA chain with longer, detailed output
    pipe = pipeline("text2text-generation", model="google/flan-t5-large", tokenizer="google/flan-t5-large", max_length=256)
    llm = HuggingFacePipeline(pipeline=pipe)
    chain = load_qa_chain(llm=llm, chain_type="stuff")

    st.success(" PDF loaded successfully. Ask your question!")

    query = st.text_input("Ask a question from the PDF:")
    if query:
        docs = vectorstore.similarity_search(query, k=3)
        response = chain.run(input_documents=docs, question=query)
        st.markdown("**Answer:**")
        st.write(response)
