from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

embedding_model = load_model()

def emb_text(text):
    return embedding_model.encode([text],normalize_embeddings=True).tolist()[0]

def get_dimension():
    return embedding_model.get_sentence_embedding_dimension()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text