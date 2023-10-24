import streamlit as st
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import logging
from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel
from typing import Any, List
import os
import subprocess

def check_and_install_libgl():
    try:
        result = subprocess.run(["ldconfig", "-p"], text=True, capture_output=True)
        if "libGL.so.1" not in result.stdout:
            print("Installing libGL.so.1...")
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "libgl1-mesa-glx"], check=True)
    except Exception as e:
        print(f"An error occurred: {e}")

check_and_install_libgl()

logging.basicConfig(level=logging.INFO)

def load_llm(model_name):
    llm = CTransformers(
        model=model_name,
        model_type="mistral",
        max_new_tokens=1048,
        temperature=0
    )
    return llm



class Element(BaseModel):
    type: str
    text: Any

def file_processing(file_path: str) -> List[Element]:
    raw_pdf_elements = partition_pdf(
        filename=file_path,
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )

    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

    return categorized_elements

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def llm_pipeline(file_path: str, model_name: str):
    elements = file_processing(file_path)

    # Segregate text and table elements
    texts = [e.text for e in elements if e.type == "text"]
    tables = [e.text for e in elements if e.type == "table"]

    # Process text elements
    documents_text = [Document(page_content=t) for t in texts]
    embeddings_text = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store_text = FAISS.from_documents(documents_text, embeddings_text)

    # Process table elements
    documents_table = [Document(page_content=t) for t in tables]
    embeddings_table = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store_table = FAISS.from_documents(documents_table, embeddings_table)

    # Create retriever for text and table elements
    retriever_text = vector_store_text.as_retriever()
    retriever_table = vector_store_table.as_retriever()

    # Load llm and create answer generation chain
    llm_answer_gen = load_llm(model_name)
    answer_generation_chain_text = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff", retriever=retriever_text)
    answer_generation_chain_table = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff", retriever=retriever_table)

    # You might want to combine or utilize the text and table retrievers/chains accordingly in your app
    return answer_generation_chain_text, answer_generation_chain_table

# ... (rest of your imports and functions are unchanged)

def question_over_pdf_app():
    model_selection = st.sidebar.selectbox(
        'Select a Model:',
        ('TheBloke/Mistral-7B-Instruct-v0.1-GGUF', 'TheBloke/zephyr-7B-alpha-GGUF')
    )

    uploaded_file = st.sidebar.file_uploader("Upload your PDF file here", type=['pdf'])

    if uploaded_file:
        with st.spinner("Analyzing..."):  # Corrected line
            with open("temp_pdf.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            answer_generation_chain_text, answer_generation_chain_table = llm_pipeline("temp_pdf.pdf", model_selection)
        st.sidebar.success("PDF Analyzed! You can now ask questions.")

        question = st.sidebar.text_input("Posez votre question ici")

        if st.sidebar.button("Ask"):
            with st.spinner("Fetching answer..."):
                response_text = answer_generation_chain_text.run(question)
                response_table = answer_generation_chain_table.run(question)
                # Here you might want to decide how to handle the responses from text and table chains
                # For simplicity, we'll just show both responses
                st.write(f"Text Response: {response_text}")
                st.write(f"Table Response: {response_table}")


st.set_page_config(page_title="Question over PDF using HF", page_icon="📖")
st.markdown("# Question over text and table using HF")
st.sidebar.header("Question over text and table using HF")
st.write(
    """This app allows you to upload a PDF, select a model, and ask questions based on the content of the text and table."""
)

question_over_pdf_app()


