# 📚 Intelligent Document Q&A System

A powerful Retrieval-Augmented Generation (RAG) application built with Streamlit that allows users to upload PDF documents and ask questions using natural language. The system leverages advanced AI models to provide accurate, context-aware answers from your documents.

## ✨ Features

- **PDF Document Processing**: Upload and process multiple PDF files simultaneously
- **Intelligent Question Answering**: Ask questions in natural language and get accurate responses
- **Vector Database**: Efficient document retrieval using FAISS vector store
- **Chat History**: Keep track of all your questions and answers
- **Customizable Settings**: Adjust chunk size and overlap for optimal performance
- **Modern UI**: Clean, responsive interface with gradient styling
- **Real-time Processing**: Live progress indicators for document processing
- **Session Management**: Clear chat history or reset the entire system

## 🚀 Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: Framework for developing LLM applications
- **Groq**: Fast LLM inference with Llama 3.1
- **HuggingFace**: Embeddings model (all-MiniLM-L6-v2)
- **FAISS**: Vector similarity search
- **PyPDF**: PDF document loading and parsing

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key
- HuggingFace token (optional, for private models)

## 🔧 Installation

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd document-qa-system
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 📦 Dependencies

Create a `requirements.txt` file with the following:

```txt
streamlit==1.31.0
langchain==0.1.0
langchain-groq==0.0.1
langchain-huggingface==0.0.1
langchain-community==0.0.13
faiss-cpu==1.7.4
pypdf==3.17.4
python-dotenv==1.0.0
sentence-transformers==2.3.1
```

## 🎮 Usage

1. **Start the application**
```bash
streamlit run app.py
```

2. **Upload Documents**
   - Click "Browse files" to select one or more PDF files
   - Click "🚀 Process Documents" to create the vector database

3. **Ask Questions**
   - Enter your question in the text input field
   - Click "💬 Ask Question" or press Enter
   - View the answer in the conversation history

4. **Adjust Settings** (Sidebar)
   - **Chunk Size**: Control how documents are split (500-2000 characters)
   - **Chunk Overlap**: Set overlap between chunks (50-500 characters)
   - **Clear Chat History**: Remove all previous Q&A pairs
   - **Reset All**: Clear everything and start fresh

## 🏗️ Project Structure

```
document-qa-system/
├── app.py                 # Main application file
├── .env                   # Environment variables (not in repo)
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore file
```

## 🎨 Features Breakdown

### Document Processing
- Supports multiple PDF uploads
- Automatic text extraction and chunking
- Progress tracking with visual feedback
- Temporary file handling for security

### Vector Database
- FAISS for efficient similarity search
- Customizable chunking parameters
- Persistent session state
- Retrieval of top-k relevant documents

### Question Answering
- Context-aware responses
- Accurate source attribution
- Conversation history tracking
- Timestamp for each interaction

## 🔐 API Keys

### Getting a Groq API Key
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy and add to your `.env` file

### Getting a HuggingFace Token (Optional)
1. Visit [HuggingFace](https://huggingface.co/)
2. Sign up or log in
3. Go to Settings → Access Tokens
4. Create a new token
5. Copy and add to your `.env` file

## ⚙️ Configuration

### Model Settings
- **LLM Model**: `llama-3.1-8b-instant` (Groq)
- **Embedding Model**: `all-MiniLM-L6-v2` (HuggingFace)
- **Retrieval**: Top 3 relevant chunks

### Default Parameters
- Chunk Size: 1000 characters
- Chunk Overlap: 200 characters
- Search Results: 3 documents

## 🐛 Troubleshooting

**Issue**: "API key not found"
- **Solution**: Ensure your `.env` file contains valid API keys

**Issue**: "Module not found"
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: "PDF processing failed"
- **Solution**: Ensure PDF is not corrupted or password-protected

**Issue**: "Out of memory"
- **Solution**: Reduce chunk size or process fewer documents at once

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain for the RAG framework
- Groq for lightning-fast LLM inference
- HuggingFace for embedding models
- Streamlit for the web framework

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Made with ❤️ using Streamlit and LangChain**
