import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Ensure the directory exists
upload_dir = "./uploaded_files"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

def vector_embedding(file_path):
    if "vectors" not in st.session_state:
        # Document loading from the uploaded file
        st.session_state.loader = PyPDFDirectoryLoader(upload_dir)  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading

        # Display document metadata
        st.session_state.doc_info = {
            "Number of Pages": len(st.session_state.docs),
            "File Name": os.path.basename(file_path)
        }

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=50)  # Chunk Creation

        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:30])  # Splitting

        # Vector embeddings
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings)

        st.session_state.processed = True  # Mark the file as processed

st.title("Nvidia NIM Demo")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    if "processed" not in st.session_state:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Processing file..."):
            vector_embedding(file_path)
        st.success("File uploaded and processed successfully!")
        st.write("Document Info:", st.session_state.doc_info)
    else:
        st.warning("File has already been processed. Please enter your question.")

prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
else:
    st.write("Please upload a PDF file and enter a question.")
