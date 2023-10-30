import streamlit as st
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import logging

logging.basicConfig(level=logging.INFO)

def load_llm(model_name):
    llm = CTransformers(
        model=model_name,
        model_type="mistral",
        max_new_tokens=1048,
        temperature=0
    )
    return llm

def file_processing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    content = ''
    for page in data:
        content += page.page_content
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    chunks = splitter.split_text(content)
    documents = [Document(page_content=t) for t in chunks]
    return documents

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def llm_pipeline(file_path, model_name):
    documents = file_processing(file_path)
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vector_store = FAISS.from_documents(documents, embeddings)
    llm_answer_gen = load_llm(model_name)
    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                                          chain_type="stuff", 
                                                          retriever=vector_store.as_retriever())
    return answer_generation_chain

def question_over_pdf_app():
    model_selection = st.sidebar.selectbox(
        'Select a Model:',
        ('TheBloke/Mistral-7B-Instruct-v0.1-GGUF', 'TheBloke/zephyr-7B-alpha-GGUF','TheBloke/zephyr-7B-beta-GGUF')
    )

    uploaded_file = st.sidebar.file_uploader("Upload your PDF file here", type=['pdf'])

    if uploaded_file:
        with st.spinner("Analyzing..."):
            with open("temp_pdf.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            answer_generation_chain = llm_pipeline("temp_pdf.pdf", model_selection)
        st.sidebar.success("PDF Analyzed! You can now ask questions.")

        question = st.sidebar.text_input("Posez votre question ici")

        if st.sidebar.button("Ask"):
            with st.spinner("Fetching answer..."):
                response = answer_generation_chain.run(question)
                st.write(response)

st.set_page_config(page_title="Question over PDF using HF", page_icon="📖")
st.markdown("# Question over PDF using HF")
st.sidebar.header("Question over PDF using HF")
st.write(
    """This app allows you to upload a PDF, select a model, and ask questions based on the content of the PDF."""
)

question_over_pdf_app()

# You can call show_code if you want to show the code of this app
# show_code(question_over_pdf_app)
