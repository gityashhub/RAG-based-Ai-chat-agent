import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
#                        CUSTOM CSS (UNCHANGED)
# -------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #ddd;
        margin-bottom: 2rem;
    }
    .chat-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .user-message {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        color: #000;
    }
    .bot-message {
        background: #f1f8e9;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
        color: #000;
    }
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeeba;
    }
    .metric-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #444;
    }
    .metric-item:last-child {
        border-bottom: none;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #fff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.2rem;
    }
    .section-header {
        margin-bottom: 1rem;
    }
    .section-header h4 {
        margin: 0 0 0.3rem 0;
        color: #495057;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .section-header p {
        margin: 0;
        color: #6c757d;
        font-size: 0.9rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------------
#                                HEADER
# -------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🤖 Intelligent Document Q&A System</h1>
    <p>Upload your PDFs and ask questions powered by AI</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
#                               SIDEBAR
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("System Status")
    st.markdown(f"""
    <div class="metric-item">
        <div class="metric-value">{len(st.session_state.processed_files)}</div>
        <div class="metric-label">Files Processed</div>
    </div>
    <div class="metric-item">
        <div class="metric-value">{len(st.session_state.chat_history)}</div>
        <div class="metric-label">Questions Asked</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.processed_files:
        st.subheader("Processed Files")
        for file in st.session_state.processed_files:
            st.write(f"✅ {file}")

    st.markdown("---")

    st.subheader("Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Reset All"):
        st.session_state.vectorstore = None
        st.session_state.processed_files = []
        st.session_state.chat_history = []
        st.rerun()

# -------------------------------------------------------------------------
#                            LAYOUT COLUMNS
# -------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

# ---------------------- UPLOAD SECTION ----------------------
with col1:
    st.markdown("""
    <div class="section-header">
        <h4>Upload Documents</h4>
        <p>Upload your PDF files to create a knowledge base</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size / 1024:.1f} KB)")

    process_button = st.button("🚀 Process Documents", use_container_width=True)

# ---------------------- QUESTION SECTION ----------------------
with col2:
    st.markdown("""
    <div class="section-header">
        <h4>Ask Questions</h4>
        <p>Query your documents using natural language</p>
    </div>
    """, unsafe_allow_html=True)

    user_prompt = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic of the document?"
    )

    ask_button = st.button("💬 Ask Question", use_container_width=True)

# -------------------------------------------------------------------------
#                     PROCESS DOCUMENTS BUTTON
# -------------------------------------------------------------------------
if process_button:
    if not uploaded_files:
        st.markdown("""
        <div class="status-warning">
            ⚠️ Please upload at least one PDF file before processing.
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("🔄 Processing documents..."):
            try:
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name="llama-3.1-8b-instant"
                )

                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                all_documents = []
                processed_files = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                    all_documents.extend(docs)
                    processed_files.append(uploaded_file.name)

                    progress_bar.progress((i + 1) / len(uploaded_files))
                    os.unlink(tmp_path)

                status_text.text("Creating vector database...")

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                split_docs = splitter.split_documents(all_documents)
                vectorstore = FAISS.from_documents(split_docs, embeddings)

                st.session_state.vectorstore = vectorstore
                st.session_state.processed_files = processed_files

                progress_bar.progress(1.0)
                status_text.empty()

                st.markdown(f"""
                <div class="status-success">
                    ✅ Successfully processed {len(uploaded_files)} file(s)!
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# -------------------------------------------------------------------------
#                           ASK QUESTION
# -------------------------------------------------------------------------
if ask_button or user_prompt:
    if user_prompt:
        if st.session_state.vectorstore is None:
            st.markdown("""
            <div class="status-warning">
                ⚠️ Please upload and process PDF files first.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name="llama-3.1-8b-instant"
                    )

                    prompt = ChatPromptTemplate.from_template("""
                    Answer the question strictly using the context below.
                    If the answer is not in the context, say:
                    "I could not find the answer in the document."

                    <context>
                    {context}
                    </context>

                    Question: {input}

                    Answer:
                    """)

                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

                    # Format docs into a single string
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    # LCEL runnable chain
                    chain = (
                        {
                            "context": retriever | RunnableLambda(format_docs),
                            "input": RunnablePassthrough()
                        }
                        | prompt
                        | llm
                    )

                    # Pass ONLY the question string (fixes the dict/replace bug)
                    response = chain.invoke(user_prompt)

                    final_answer = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )

                    st.session_state.chat_history.append({
                        "question": user_prompt,
                        "answer": final_answer,
                        "timestamp": time.strftime("%H:%M:%S")
                    })

                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")

# -------------------------------------------------------------------------
#                      SHOW CHAT HISTORY
# -------------------------------------------------------------------------
if st.session_state.chat_history:
    st.markdown("## Conversation History")

    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(
            f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."
            if len(chat['question']) > 50 else
            f"Q{len(st.session_state.chat_history)-i}: {chat['question']}",
            expanded=(i == 0)
        ):
            st.markdown(f"""
            <div class="user-message">
                <strong>Question ({chat['timestamp']}):</strong><br>
                {chat['question']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="bot-message">
                <strong>Answer:</strong><br>
                {chat['answer']}
            </div>
            """, unsafe_allow_html=True)
