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
import requests
from urllib.parse import urlparse
import pdfkit
import subprocess
import shlex

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

def html_to_pdf(input_html, output_pdf):
    command = f"xvfb-run wkhtmltopdf --enable-local-file-access --quiet --no-images --disable-external-links --disable-javascript {shlex.quote(input_html)} {shlex.quote(output_pdf)}"
    try:
        subprocess.check_call(command, shell=True)
        print('PDF conversion successful!')
        return output_pdf
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during PDF conversion: {e}")
        return None

    
def url_to_pdf(full_url, personal_access_token):
    # Parse the full URL to separate the domain and the path
    parsed_url = urlparse(full_url)
    domain = parsed_url.netloc
    page_path = parsed_url.path

    headers = {
        "Authorization": f"Bearer {personal_access_token}",
        "Accept": "application/json"
    }

    # Make the request to the specified URL
    response = requests.get(f'https://{domain}{page_path}', headers=headers)
    
    # Check the response
    if response.status_code == 200:
        print('Connection successful!')
        
        # Save the HTML content to a file
        with open('temp_pdf.html', 'w') as file:
            file.write(response.text)
        
        # Convert the HTML file to PDF using the html_to_pdf function
        return html_to_pdf('temp_pdf.html', 'temp_pdf.pdf')
        
    else:
        print(f'Failed to connect, status code: {response.status_code}')
        print('Response:', response.text)
        return None

def question_over_pdf_app():
    model_selection = st.sidebar.selectbox(
        'Select a Model:',
        ('TheBloke/Mistral-7B-Instruct-v0.1-GGUF', 'TheBloke/zephyr-7B-alpha-GGUF', 'TheBloke/zephyr-7B-beta-GGUF','TheBloke/neural-chat-7B-v3-1-GGUF')
    )

    uploaded_file = st.sidebar.file_uploader("Upload your PDF file here", type=['pdf'])
    file_path_input = st.sidebar.text_input("...or enter a file path here")
    confluence_url = st.sidebar.text_input("...or enter a Confluence page URL here")
    personal_access_token = st.sidebar.text_input("Enter your Confluence Personal Access Token")

    file_to_process = None

    if uploaded_file:
        file_to_process = "temp_pdf.pdf"
        with open(file_to_process, "wb") as f:
            f.write(uploaded_file.getvalue())
    elif file_path_input:
        file_to_process = file_path_input
    elif confluence_url and personal_access_token:
        # Call the url_to_pdf function and get the path to the processed PDF
        file_to_process = url_to_pdf(confluence_url, personal_access_token)

    if file_to_process:
        with st.spinner("Analyzing..."):
            answer_generation_chain_text, answer_generation_chain_table = llm_pipeline(file_to_process, model_selection)
        st.sidebar.success("PDF Analyzed! You can now ask questions.")

        question = st.sidebar.text_input("Posez votre question ici")

        if st.sidebar.button("Ask"):
            with st.spinner("Fetching answer..."):
                response_text = answer_generation_chain_text.run(question)
                response_table = answer_generation_chain_table.run(question)
                st.write(f"Text Response: {response_text}")
                st.write(f"Table Response: {response_table}")

st.set_page_config(page_title="Question over PDF using HF", page_icon="📖")
st.markdown("# Question over text and table using HF")
st.sidebar.header("Question over text and table using HF")
st.write(
    """This app allows you to upload a PDF, select a model, and ask questions based on the content of the text and table."""
)

question_over_pdf_app()


