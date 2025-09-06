from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_Chunks(text:str) ->  list[str] :
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks