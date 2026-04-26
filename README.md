# RAG-Powered Multi-Agent Q&A Assistant

## Overview

This project implements a sophisticated knowledge assistant that combines Retrieval-Augmented Generation (RAG) with a multi-agent architecture to deliver accurate, context-aware responses. The system intelligently routes queries to specialized tools based on query intent, retrieves relevant information from a document collection, and generates natural language answers using state-of-the-art language models.

**System Architecture** Intelligent query routing between specialized tools based on intent detection
- **Retrieval-Augmented Generation (RAG)**: Enhances LLM responses with relevant context from a document collection
- **Interactive Web Interface**: Clean, user-friendly Streamlit application for asking questions and viewing results
- **Document Management**: Upload and process new documents through the web interface
- **Specialized Tools**:
  - **RAG Tool**: Retrieves and synthesizes information from document collection
  - **Calculator Tool**: Performs mathematical calculations with comprehensive operator support
  - **Dictionary Tool**: Provides definitions enhanced by document context

## Technology Stack

- **Vector Database**: FAISS for efficient similarity search
- **Embeddings**: HuggingFace's E5 model (intfloat/e5-base-v2)
- **LLM Integration**: Groq API with Llama 3 models for fast, accurate responses
- **Frontend**: Streamlit for interactive web interface
- **Document Processing**: LangChain's document loaders and text splitters

## Architecture

The system follows a modular architecture with several key components:

1. **Document Ingestion Pipeline**:
   - Loads documents from the data directory
   - Splits documents into semantically meaningful chunks
   - Computes embeddings for each chunk
   - Stores vectors in a FAISS index for efficient retrieval

2. **Multi-Agent System**:
   - Analyzes query intent using keyword detection and pattern matching
   - Routes queries to specialized tools based on intent
   - Orchestrates the retrieval and generation process

3. **Response Generation**:
   - Retrieves relevant document chunks based on query similarity
   - Constructs prompts that combine the query with retrieved context
   - Generates coherent, contextually accurate responses using Groq's Llama 3 models

## Installation and Setup

### Prerequisites
- Python 3.8+
- Groq API key

### Installation

1. Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/rag-agent-qa.git](https://github.com/rajdeep-senapati/RAG-powered-QA_Assistant-.git)
    cd agent
    ```

2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. Create a `.env` file with your API keys:
    ```
    GROQ_API_KEY=your_groq_api_key_here
    ```

### Running the Application

1. Add documents to the `data` directory (sample documents are provided)

2. Ingest the documents to create the vector store:
    ```bash
    python ingest.py
    ```

3. Launch the Streamlit application:
    ```bash
    streamlit run app.py
    ```

4. Open your browser and navigate to `http://localhost:8501`

## Usage Examples

### Information Retrieval
- "What are the main factors driving electric vehicle adoption?"
- "Explain the negative impacts of uncontrolled EV charging on power systems."

### Calculations
- "Calculate 25 * 16 + 42"
- "What is the square root of 169 divided by 13?"

## Design Choices and Optimizations

- **Chunking Strategy**: 1000-character chunks with 200-character overlap balances context preservation with retrieval precision
- **E5 Embeddings**: Offers comparable performance to OpenAI embeddings with smaller vector dimensions (768 vs 1536)
- **Direct Groq Integration**: Bypasses LangChain wrappers for more reliable API interactions
- **Error Handling**: Graceful fallbacks ensure system reliability even when components fail

## Future Enhancements

- Implement conversation memory for follow-up questions
- Add support for PDF and other document formats
- Integrate additional specialized tools (e.g., image analysis, code execution)
- Implement hybrid search combining sparse and dense retrievers
- Add user authentication and document permission management

## Acknowledgments

- LangChain for document processing utilities
- HuggingFace for embedding models
- Groq for LLM API access
- FAISS for vector similarity search
- Streamlit for the web interface framework

---
