import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings # Removed
from langchain_community.embeddings import HuggingFaceEmbeddings # Added
from langchain_community.vectorstores import Chroma
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
import chromadb # Add this import
# ... other imports ...

def create_and_store_embeddings(chunks: list, persist_directory: str):
    """Creates embeddings using HuggingFaceEmbeddings and stores them in ChromaDB."""
    if not chunks:
        logging.warning("No chunks to process. Skipping embedding creation.")
        return None
        
    logging.info(f"Creating embeddings with '{EMBEDDING_MODEL_NAME}' and storing in ChromaDB...")
    embeddings_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )
    
    if os.path.exists(persist_directory):
        logging.info(f"Removing existing database at {persist_directory}")
        shutil.rmtree(persist_directory)
    
    # Explicitly create a PersistentClient
    client = chromadb.PersistentClient(path=persist_directory)

    # Get or create the collection, explicitly passing the embedding function
    # This is the crucial part that might behave differently than Chroma.from_documents
    logging.info("Getting or creating ChromaDB collection with specified embedding function...")
    collection_name = "rag_collection" # You can name your collection
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embeddings_function # Pass the HuggingFaceEmbeddings instance
            # metadata={"hnsw:space": "cosine"} # Optional: specify distance metric
        )
    except Exception as e:
        # Catching a broader exception here because the exact type might vary
        # if onnxruntime is still being problematic at this stage.
        if "onnxruntime" in str(e).lower() or "dll" in str(e).lower():
            logging.error(f"Failed to create collection even with explicit embedding function due to: {e}")
            logging.error("This might indicate that ChromaDB's import of its default ONNX embedder is happening regardless.")
            logging.error("Consider Option 2 (using FAISS) if this persists.")
            return None
        else:
            logging.error(f"An unexpected error occurred while creating collection: {e}")
            raise # Re-raise if it's not the ONNX issue

    # Prepare documents for adding to the collection
    # We need IDs and the document texts (page_content)
    # Metadata can also be included.
    documents_to_add = [chunk.page_content for chunk in chunks]
    metadatas_to_add = [chunk.metadata for chunk in chunks]
    ids_to_add = [f"chunk_{i}" for i in range(len(chunks))] # Simple unique IDs

    if not documents_to_add:
        logging.warning("No document content to add to the collection.")
        return client # Or collection, depending on what you want to return

    logging.info(f"Adding {len(documents_to_add)} chunks to the collection '{collection_name}'...")
    try:
        collection.add(
            documents=documents_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        logging.info(f"Embeddings stored successfully in {persist_directory} in collection '{collection_name}'.")
    except Exception as e:
        logging.error(f"Failed to add documents to Chroma collection: {e}")
        return None # Or raise

    # For LangChain compatibility, Chroma.from_documents returns a LangChain VectorStore wrapper.
    # We can wrap our manually created Chroma setup similarly if needed by the agent.
    # However, for ingestion, just ensuring data is stored is key.
    # For the agent.py, you'd initialize Chroma like:
    # vector_store = Chroma(
    #     client=client, # or client_settings=chromadb.config.Settings(...) if path based
    #     collection_name=collection_name,
    #     embedding_function=embeddings_function,
    #     persist_directory=persist_directory # Important for loading
    # )
    # For now, let's return the client or indicate success.
    # The LangChain Chroma wrapper can be instantiated in agent.py when loading.
    return client # Or simply True to indicate success

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # No longer needed for ingestion
# if not OPENAI_API_KEY:
#     logging.error("OPENAI_API_KEY not found in .env file or environment variables.")
#     raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

# Define constants
DATA_PATH = "data/"
PERSIST_DIRECTORY = "db_chroma"
# Define the embedding model name
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A good default, runs locally

def load_documents(data_path: str):
    """Loads all .txt and .pdf documents from the specified directory."""
    logging.info(f"Loading documents from {data_path}...")
    
    all_docs = []
    doc_files = os.listdir(data_path)
    
    if not doc_files:
        logging.warning(f"No files found in {data_path}. Please add your documents.")
        return []

    for doc_file in doc_files:
        file_path = os.path.join(data_path, doc_file)
        try:
            if doc_file.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
                all_docs.extend(loader.load())
                logging.info(f"Successfully loaded TXT file: {doc_file}")
            elif doc_file.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                all_docs.extend(loader.load_and_split())
                logging.info(f"Successfully loaded PDF file: {doc_file}")
            else:
                logging.info(f"Skipping unsupported file type: {doc_file}")
        except Exception as e:
            logging.error(f"Error loading {doc_file}: {e}")
    
    if not all_docs:
        logging.warning(f"No processable (.txt or .pdf) documents were loaded from {data_path}.")
        return []

    logging.info(f"Loaded {len(all_docs)} document pages/sections in total.")
    return all_docs

def chunk_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Splits documents into smaller chunks."""
    if not documents:
        logging.warning("No documents provided for chunking.")
        return []
        
    logging.info("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

def create_and_store_embeddings(chunks: list, persist_directory: str):
    """Creates embeddings using HuggingFaceEmbeddings and stores them in ChromaDB."""
    if not chunks:
        logging.warning("No chunks to process. Skipping embedding creation.")
        return None
        
    logging.info(f"Creating embeddings with '{EMBEDDING_MODEL_NAME}' and storing in ChromaDB...")
    # Initialize HuggingFace Embeddings
    # model_kwargs = {'device': 'cpu'} # Explicitly use CPU if needed, or let it auto-detect
    # encode_kwargs = {'normalize_embeddings': False} # Depending on the model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # model_kwargs=model_kwargs,
        # encode_kwargs=encode_kwargs
    )
    
    if os.path.exists(persist_directory):
        logging.info(f"Removing existing database at {persist_directory}")
        shutil.rmtree(persist_directory)
        
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    logging.info(f"Embeddings stored successfully in {persist_directory}.")
    return vector_store

def main():
    logging.info("Starting data ingestion process...")
    
    documents = load_documents(DATA_PATH)
    if not documents:
        logging.info("No documents loaded. Exiting ingestion process.")
        return

    text_chunks = chunk_documents(documents)
    if not text_chunks:
        logging.info("No text chunks created. Exiting ingestion process.")
        return
        
    vector_store = create_and_store_embeddings(text_chunks, PERSIST_DIRECTORY)
    if vector_store:
        logging.info("Data ingestion process completed successfully.")
    else:
        logging.warning("Data ingestion process completed, but no vector store was created.")

if __name__ == "__main__":
    main()
