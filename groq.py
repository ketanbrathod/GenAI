import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

# Load .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq Model
GROQ_MODEL = "llama-3.3-70b-versatile"

# Check API Key
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

st.header("My First RAG Application")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(
        "Upload a PDF file and start asking questions",
        type="pdf"
    )

if file is not None:

    # Read PDF
    pdf_reader = PdfReader(file)

    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User Question
    user_question = st.text_input("Type your question here")

    if user_question:

        # Similarity Search
        match = vector_store.similarity_search(user_question)

        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.1
        )

        # QA Chain (uses standard default prompt built into load_qa_chain)
        chain = load_qa_chain(
            llm,
            chain_type="stuff"
        )

        # Response
        response = chain.invoke(
            {
                "input_documents": match,
                "question": user_question
            }
        )

        st.write(response["output_text"])