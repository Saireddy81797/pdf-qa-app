import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="📄 PDF Q&A App (Fast & Stable)", layout="centered")
st.title("📄 Ask Questions from PDF (Fast & Light)")

with st.sidebar:
    openai_api_key = st.text_input("🔐 OpenAI API Key", type="password")
    st.markdown("[Get your OpenAI key](https://platform.openai.com/account/api-keys)")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

uploaded_file = st.file_uploader("📤 Upload PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    # Split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text or "")

    query = st.text_input("💬 Ask a question from the PDF:")
    if query:
        with st.spinner("Thinking..."):
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.2)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            # Give full context to LLM
            docs = [{"page_content": chunk} for chunk in chunks]
            answer = chain.run(input_documents=docs, question=query)
            st.success("Answer:")
            st.write(answer)
