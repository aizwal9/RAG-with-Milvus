# Milvus RAG

This project is a Retrieval-Augmented Generation (RAG) application that uses Milvus as a vector database to answer questions about uploaded PDF documents. The application is built with Streamlit and uses a large language model to generate responses.

![image](https://github.com/aizwal9/RAG-with-Milvus/blob/main/img.png)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the environment variables:**
   Create a `.env` file in the root directory of the project and add the following environment variables:
   ```
   MILVUS_HOST="localhost"
   MILVUS_PORT="19530"
   ```

## Usage

1. **Run the Streamlit application:**
   ```bash
   streamlit run main.py
   ```

2. **Upload PDF files:**
   Use the sidebar in the Streamlit application to upload one or more PDF files.

3. **Process PDFs:**
   Click the "Process PDFs" button to process the uploaded PDF files and store them in the Milvus vector database.

4. **Ask questions:**
   Ask questions related to the content of the PDF files in the chat input box.

## Dependencies

The following dependencies are required to run the project:

- streamlit
- ollama
- sentence-transformers
- langchain==0.3.27
- langchain-community==0.3.27
- langchain-core==0.3.72
- langchain-openai==0.3.28
- langchain-text-splitters==0.3.9
- langgraph==0.5.4
- langgraph-checkpoint==2.1.1
- langgraph-prebuilt==0.5.2
- langgraph-sdk==0.1.74
- langsmith==0.4.8
- pypdf
