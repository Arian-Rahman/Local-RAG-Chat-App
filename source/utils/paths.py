import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
print(f"Project root directory: {project_root}")

artifacts_dir = os.path.join(project_root,"artifacts")
models_dir = os.path.join(project_root,"models")

paths = {
    "pdfs" : os.path.join(project_root, "pdf"),
    "extracted_data": os.path.join(artifacts_dir, "extracted_data.txt"),
    "embedding_nomic": os.path.join(models_dir, "nomic-embed-text-v2-moe"),
    "embedding_baai": os.path.join(models_dir, "llm-embedder"),
    "embedding_save": os.path.join(artifacts_dir, "embeddings.pt"),
    "faiss_index": os.path.join(artifacts_dir, "faiss_index.index"),
    "chunks_save": os.path.join(artifacts_dir, "split_chunks.json"),
    "index_to_metadata": os.path.join(project_root, "artifacts", "index_to_metadata.json"),

}

