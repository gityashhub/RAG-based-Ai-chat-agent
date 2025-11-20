import streamlit as st
import os
import time

from dotenv import load_dotenv
load_dotenv()

# Set API Keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# ----------------------------- LangChain Imports (NEW API) -----------------------------

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# combine docs chain moved to community package
from langchain_community.chains.combine_documents import create_stuff_documents_chain

# retrieval now uses RunnableSequence
from langchain_core.runnables import RunnablePassthrough

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# ---------------------------------------------------------------------------------------


# LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="Llama3-8b-8192"
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question strictly from the given context.
If the answer is not found, say "I don't know".

<context>
{context}
</context>

Question: {question}
""")

# Vector embedding creation
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")

        # Load documents
        st.session_state.docs = st.session_state.loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = splitter.split_documents(st.session_state.docs[:50])

        # Vectorstore
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )


st.title("RAG Document Q&A With Groq + Llama3")

user_prompt = st.text_input("Ask from your research papers:")

if st.button("Create Vector Embeddings"):
    create_vector_embedding()
    st.success("Vector database created successfully!")

if user_prompt:
    # Document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retriever
    retriever = st.session_state.vectors.as_retriever()

    # New style retrieval chain
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | document_chain
    )

    start = time.process_time()
    response = retrieval_chain.invoke({"question": user_prompt})
    print("Response time:", time.process_time() - start)

    # Output
    st.write("### Answer:")
    st.write(response)

    with st.expander("🔎 Similar Documents"):
        docs = retriever.get_relevant_documents(user_prompt)
        for d in docs:
            st.write(d.page_content)
            st.write("-----")
