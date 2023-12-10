import streamlit as st
import logging
from typing import Any, List
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client import models
import io

# Qdrant client setup
client = qdrant_client.QdrantClient(
    "https://5321c90b-0709-43e9-8081-e899e9bc9e94.us-east4-0.gcp.cloud.qdrant.io",
    api_key="eDt8YdkRqa-hdEigsI1C6yV6XFXdThHBcdIcsNoHrUFTqWCByrmn4g"
)

# Define vector configuration with cosine distance
vector_params = models.VectorParams(size=768, distance=models.Distance.COSINE)

# Check if the collection exists and create it if it does not
existing_collections = client.get_collections().collections
if "my_collection" not in [col.name for col in existing_collections]:
    client.create_collection("my_collection", vectors_config=vector_params)

# Create the index
index = QdrantVectorStore(client=client, collection_name="my_collection")

logging.basicConfig(level=logging.INFO)

class Element(BaseModel):
    type: str
    text: Any

def file_processing(file_like_object) -> List[Element]:
    # Convert Streamlit uploaded file to a file-like object
    file_stream = io.BytesIO(file_like_object.getbuffer())

    raw_pdf_elements = partition_pdf(
        file=file_stream,  # Use the file parameter
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

def process_pdf(file_like_object):
    elements = file_processing(file_like_object)
    model_name = 'sentence-transformers/all-mpnet-base-v2'  # Embedding model
    embeddings_model = HuggingFaceEmbeddings()
    
    documents = [Document(page_content=e.text) for e in elements]
    for doc in documents:
        try:
            # Use embed_query or embed_documents based on your requirement
            embedding = embeddings_model.embed_query(doc.page_content) 
            # Or for multiple documents: embeddings_model.embed_documents([doc.page_content])
            payload = {'content': doc.page_content}
            index.add_documents([embedding], payload=[payload])
        except Exception as e:
            logging.error(f"Error in embedding document: {e}")

    summary_message = f"Processed {len(documents)} elements"
    preview_content = "\n".join([doc.page_content[:100] + "..." for doc in documents])
    return summary_message, preview_content

# Streamlit UI
st.title('PDF Ingestion Pipeline')
uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)

if st.button('Process and Store in Qdrant'):
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        summary_message, preview_content = process_pdf(uploaded_file)
        st.success(summary_message)
        st.text("Content Preview:")
        st.write(preview_content)
