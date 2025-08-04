import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Custom embedding class (fix for Streamlit Cloud)
class LocalEmbedding(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=False).tolist()

# Load QA chain using Flan-T5
@st.cache_resource
def load_qa_chain_model():
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base", max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    return load_qa_chain(llm=llm, chain_type="stuff")

# Streamlit App UI
st.set_page_config(page_title=" PDF QA App", layout="centered")
st.title(" Ask Questions from PDF (Offline, No API Key)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    if not raw_text.strip():
        st.warning("PDF has no extractable text.")
    else:
        # Split the text
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=150)
        texts = splitter.split_text(raw_text)

        # Embed and store in vector DB
        embeddings = LocalEmbedding()
        db = FAISS.from_texts(texts, embedding=embeddings)

        st.success(" PDF processed. Now ask a question.")

        # Load model
        chain = load_qa_chain_model()

        query = st.text_input("Ask a question from the PDF:")
        if query:
            docs = db.similarity_search(query, k=3)
            response = chain.run(input_documents=docs, question=query)
            if len(response.split()) < 30:
                st.info(" Answer is short. Consider rephrasing your question or uploading a clearer PDF.")
            st.markdown("###  Answer:")
            st.write(response)
