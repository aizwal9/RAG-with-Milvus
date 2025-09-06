import ollama
import streamlit as st

from milvus import load_data_into_milvus, query
from retriever import get_Chunks
from util import get_pdf_text, get_dimension


def get_rag_response(question: str) -> str:
    search_res = query(question)
    retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
    context = "\n".join([line_with_distances[0] for line_with_distances in retrieved_lines_with_distances])

    prompt: str = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """

    messages =[
        {
            'role': 'system',
            'content': 'You are a helpful assistant. Do not include any internal thoughts or reasoning in your responses. Exclude the <think> section from final answer. Only provide the final answer.'
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    print(messages)
    response = ollama.chat(model='qwen3:8b', messages=messages,think=False)

    return response['message']['content']


def main():
    st.title("Milvus RAG")
    st.set_page_config(page_title="Milvus RAG", page_icon='🤖', layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click 'Process'",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_Chunks(raw_text)

                st.session_state.vector_store = load_data_into_milvus(text_chunks, get_dimension())
                st.success("PDFs processed successfully")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question related to PDF..."):

        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.vector_store:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_rag_response(prompt)
                    st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.chat_message("assistant"):
                st.error("Please upload a PDF file First!")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Please upload and process a PRF file first!"})


if __name__ == "__main__":
    main()
