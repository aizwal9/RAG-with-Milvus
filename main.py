import ollama

from milvus import load_data_into_milvus, query
from retriever import get_Chunks
from util import get_dimension
import json

chunks = get_Chunks()
load_data_into_milvus(chunks,get_dimension())

question = "What is the legal basis for the proposal?"
search_res = query(question)

retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
print(json.dumps(retrieved_lines_with_distances, indent=4))

context = "\n".join([line_with_distances[0] for line_with_distances in retrieved_lines_with_distances])

prompt =f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

response = ollama.chat(model='qwen3:8b',messages=[
    {
        "role":"user",
        "content": prompt
    }
])

print(response['message']['content'])