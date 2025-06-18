from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=512, chunk_overlap=52):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)