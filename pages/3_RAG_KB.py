import streamlit as st
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.vector_stores import VectorStoreQuery

# Qdrant client setup
client = qdrant_client.QdrantClient(
    "https://5321c90b-0709-43e9-8081-e899e9bc9e94.us-east4-0.gcp.cloud.qdrant.io",
    api_key="eDt8YdkRqa-hdEigsI1C6yV6XFXdThHBcdIcsNoHrUFTqWCByrmn4g"
)
index = QdrantVectorStore(client=client, collection_name="my_collection")

def load_llm(model_name):
    llm = CTransformers(
        model=model_name,
        model_type="mistral",
        max_new_tokens=1048,
        temperature=0
    )
    return llm



def query_qdrant_for_documents(query, model_name):
    # Convert query to embedding
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    query_embedding = embeddings_model.embed_query(query)

    # Query Qdrant
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=10, mode="default"
    )
    query_result = index.query(vector_store_query)

    # Process query results
    documents = [result.payload['content'] for result in query_result]
    return documents

def generate_answers_from_documents(documents, llm, query):
    answers = []
    for document_content in documents:
        context_query = f"Context: {document_content}\nQuestion: {query}"
        answer = llm.generate_answer(context_query)
        answers.append(answer)

    # Handle cases where no answers are generated
    if not answers:
        answers.append("No relevant information found.")

    return answers




def question_over_pdf_app():
    model_selection = st.sidebar.selectbox(
        'Select a Model:',
        ('TheBloke/Mistral-7B-Instruct-v0.1-GGUF', 'TheBloke/zephyr-7B-alpha-GGUF','TheBloke/zephyr-7B-beta-GGUF')
    )

    question = st.sidebar.text_input("Posez votre question ici")

    if st.sidebar.button("Ask"):
        with st.spinner("Fetching answer..."):
            document_ids = query_qdrant_for_documents(question, model_selection)
            llm = load_llm(model_selection)
            answers = generate_answers_from_documents(document_ids, llm, question)
            for answer in answers:
                st.write(answer)

st.set_page_config(page_title="Question over PDF using HF", page_icon="📖")
st.markdown("# Question over text and table using HF")
st.sidebar.header("Question over text and table using HF")
question_over_pdf_app()
