from pymilvus import MilvusClient
from tqdm import tqdm

from util import emb_text

milvus_client = MilvusClient(uri="./hf_milvus_demo.db")

collections_name = "rag_collection"

def load_data_into_milvus(text_lines : list[str] ,embedding_dimensions:int):
    if milvus_client.has_collection(collections_name):
        milvus_client.drop_collection(collections_name)

    milvus_client.create_collection(
        collections_name,
        dimension=embedding_dimensions,
        metric_type="IP", # Inner Product Distance
        consistency_level = "Strong" # String consistency level
    )

    data = []
    for i,line in enumerate(tqdm(text_lines,desc="Creating embeddings")):
        data.append({"id":i,"vector":emb_text(line),"text":line})

    insert_res = milvus_client.insert(collections_name,data)
    print(insert_res["insert_count"])

def query(question:str) -> list:
    return milvus_client.search(
        collections_name,
        data = [emb_text(question)],
        limit=3,
        search_params={"metric_type":"IP","params":{}},
        output_fields=["text"]
    )