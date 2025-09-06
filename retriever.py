from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_Chunks(file_path:str = "The-AI-Act.pdf") ->  list[str] :
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    text_lines = [chunk.page_content for chunk in chunks]
    print(len(text_lines))
    return text_lines