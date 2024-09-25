from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

st.title("RAG document Q&A with GROQ and gemma:2b")

## Load the GROQ API key
os.environ["GROP_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

## Define llm
llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

## Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings(model="gemma:2b")
        st.session_state.loader=PyPDFDirectoryLoader("research_papers") # Data ingestion step
        st.session_state.docs=st.session_state.loader.load() # document loading
        st.session_state.splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents=st.session_state.splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

user_prompt = st.text_input("Enter your query from the research papers:")

if st.button("Document embedding"):
    create_vector_embedding()
    st.write("Your database is ready")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    print(f"Response time: {time.process_time()-start}")
    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document similarity search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------")
