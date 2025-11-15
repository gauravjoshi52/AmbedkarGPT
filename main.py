import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "true" 
os.environ["CHROMA_OTEL_ENDPOINT"] = "" 
from chromadb import Client
import chromadb

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def main():
    print("Starting AmbedkarGPT Q&A System...")
    
    # Step 1: Load text file
    print("Loading speech text...")
    loader = TextLoader("speech.txt")
    documents = loader.load()
    
    # Step 2: Split text into chunks
    print("Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Step 3: Create embeddings and store in ChromaDB
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Creating ChromaDB vector store...")
    vector_store = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="chroma_db")
    
    # Step 4 & 5: Setup Q&A with Mistral
    print("Initializing Mistral 7B...")
    llm = Ollama(model="mistral")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2})
    )
    
    print("System is ready! Ask questions about this speech.")
    
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == 'quit':
            break
        answer = qa_chain.run(question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()