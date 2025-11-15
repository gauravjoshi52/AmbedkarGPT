AmbedkarGPT - Q&A System
A simple command-line Q&A system that answers questions based on Dr. B.R. Ambedkar's speech using a RAG (Retrieval Augmented Generation) pipeline.

Overview:
This project implements a question-answering system that processes Dr. B.R. Ambedkar's speech text and provides answers based solely on the content of the speech. The system uses modern AI techniques including vector embeddings and local language models.

Features-
Text Processing: Loads and splits speech text into manageable chunks

Vector Embeddings: Creates semantic embeddings using HuggingFace models

Local Vector Database: Stores and retrieves information using ChromaDB

Local AI Model: Uses Mistral 7B via Ollama for answer generation

Interactive Q&A: Command-line interface for asking questions

Technical Stack:
Framework: LangChain

Vector Database: ChromaDB

Embeddings: sentence-transformers/all-MiniLM-L6-v2

LLM: Mistral 7B via Ollama

Language: Python 3.8+

Project Structure:
AmbedkarGPT/
|--- main.py              # Main code
|--- requirements.txt     # Python dependencies
|--- speech.txt           # Dr. Ambedkar's speech
|--- README.md             # This file

Installation & Setup:
Prerequisites-
Install Python 3.8+ from python.org


Install Ollama from ollama.ai :
Step-by-Step Setup-

Install Ollama and Download Mistral 7B:
# Download Ollama from https://ollama.ai/
# Then run in cmd:
ollama pull mistral

Install Python Dependencies:
pip install -r requirements.txt
pip install sentence-transformers

Usage:
To Start the Application Run:
python main.py


Ask Questions about the speech when prompted:
Example Questions:

"What is the real remedy according to the speech?"

"What does the speech say about social reform?"

"Why are shastras important in this context?"

"What is compared to a gardener in the speech?"

Exit the Application by typing quit, exit, or q


How It Works:
The system follows these steps:

Load Text: Reads the speech from speech.txt

Split Text: Divides the content into manageable chunks

Create Embeddings: Generates vector representations using HuggingFace models

Store Vectors: Saves embeddings in ChromaDB for fast retrieval

Retrieve & Answer: Finds relevant text chunks and generates answers using Mistral 7B

Notes:
The system answers questions based only on the provided speech text.

All processing happens locally - no API keys or external services required

First run may take longer as models are downloaded and initialized.