from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def emb_text(text):
    return embedding_model.encode([text],normalize_embeddings=True).tolist()[0]

def get_dimension():
    return embedding_model.get_sentence_embedding_dimension()