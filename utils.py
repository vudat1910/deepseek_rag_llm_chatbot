import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def process_documents(pdfs):
    """
    Process PDF documents through loading, splitting, and embedding.
    Returns vector store instance.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_paths = []
        for pdf in pdfs:
            path = os.path.join(temp_dir, pdf.name)
            with open(path, "wb") as f:
                f.write(pdf.getbuffer())
            pdf_paths.append(path)
        
        documents = []
        for path in pdf_paths:
            loader = PDFPlumberLoader(path)
            documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  
            chunk_overlap=150  
        )
        splits = text_splitter.split_documents(documents)
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        return vector_store

def get_retriever():
    """Initialize and return the vector store retriever"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    try:
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )

        return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})

    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None

