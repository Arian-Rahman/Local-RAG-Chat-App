import os
from sentence_transformers import SentenceTransformer
import torch
import logging
import faiss
import json
from source.utils.device import get_device
from source.utils.embedder import get_embedding_model
from source.utils.paths import paths
from source.utils.splitter import split_text

# from utils.device import get_device
# from utils.embedder import get_embedding_model
# from utils.paths import paths
# from utils.splitter import split_text


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_embeddings(embedding_save_path):
    """
    Loads pre-computed embeddings 
    """
    embeddings = torch.load(embedding_save_path,weights_only=False)
    if embeddings is None:
        logger.error("Embeddings are None. Please generate embeddings first.")
        raise ValueError("Embeddings are None. Please generate embeddings first.")
    logger.info(f"Loaded embeddings from {embedding_save_path}")
    embedding_dim = embeddings.shape[1]  
    # logger.info(f"Embedding dimension: {embedding_dim}")
    # logger.info(f"Number of embeddings: {embeddings.shape[0]}")
    # logger.info(f"Embedding dtype: {embeddings.dtype}")
    return embeddings, embedding_dim



def save_faiss_index(index, faiss_index_path):
    """
    Saves the FAISS index to a file.
    """
    if os.path.exists(faiss_index_path):
        logger.warning(f"FAISS index file already exists at {faiss_index_path}. Overwriting it.")
    else:
        logger.info(f"Saving FAISS index to {faiss_index_path}.")
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)  
    faiss.write_index(index, faiss_index_path)
    logger.info(f"FAISS index saved to {faiss_index_path}")


def generate_faiss_index():
    """
    Generates a FAISS index & stores 'index - text' mapping.
    """
    embeddings, embedding_dim = load_embeddings(paths["embedding_save"])
    
    with open(paths["chunks_save"], "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    if len(chunks) != embeddings.shape[0]:
        logger.error(f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings")
        raise ValueError(f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings")

    #index = faiss.IndexFlatL2(embedding_dim)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    logger.info(f"FAISS index created with {embeddings.shape[0]} embeddings.")

 
    save_faiss_index(index, paths["faiss_index"])

    # Save index - chunk text mapping
    index_to_metadata = {i: chunk for i, chunk in enumerate(chunks)}
    with open(paths["index_to_metadata"], "w", encoding="utf-8") as f:
        json.dump(index_to_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved index-to-text mapping to {paths['index_to_metadata']}")
        
    logger.info("FAISS index generation completed successfully.")


def generate_embedding(text,embedding_model):
    """
    Generates an embedding for a single text input.
    """
    # if not os.path.exists(embedding_model):
    #     logger.error(f"The embedding model {embedding_model} does not exist.")
    #     raise FileNotFoundError(f"The embedding model {embedding_model} does not exist.")
    
    chunks = split_text(text)

    model = embedding_model
    device= get_device()
    logger.info(f"Using device: {device}")
    
    embedding = model.encode(
        chunks,
        convert_to_tensor=False, 
        batch_size=16,
        show_progress_bar=True,
        device=device,
        normalize_embeddings=True  
    ).astype('float32') 
    logger.info(f"Successfully generated embedding for single text input.")  
    return embedding  


def generate_embeddings(file_path, embedding_model_path,embedding_save_path = paths["embedding_save"]):
    """
    Generates embeddings for text data from a file.
    The text is split into chunks, and embeddings are generated for each chunk.
    """
    if not os.path.exists(file_path):
        logger.error(f"The file {file_path} does not exist.")
        raise FileNotFoundError(f"The file {file_path} does not exist.")
        
    if not os.path.exists(embedding_model_path):
        logger.error(f"The embedding model {embedding_model_path} does not exist.")
        raise FileNotFoundError(f"The embedding model {embedding_model_path} does not exist.")

    
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    
    chunks = split_text(data)

    model = get_embedding_model()
    device = get_device() 
    logger.info(f"Using device: {device}")
    
    embeddings = model.encode(
        chunks,
        convert_to_tensor=False, # Returns numpy arrays instead of tensors
        batch_size=16,
        show_progress_bar=True,
        device=device,
        normalize_embeddings=True  # for cosine similarity
    ).astype('float32') 
    
    if os.path.exists(embedding_save_path):
        logger.warning(f"Overwriting existing embedding at {embedding_save_path}")
        
    torch.save(embeddings, embedding_save_path)
    logger.info(f"Successfully Saved Embeddings to {embedding_save_path}")
    
    
    with open(paths["chunks_save"], "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully Saved Chunks  to {paths['chunks_save']}")
        
    return chunks, embeddings


def index_search(query,k=10,embedding_model=None):
    """
    Searches the FAISS index for the top k nearest neighbors of a query embedding.
    """
    _, embedding_dim = load_embeddings(paths["embedding_save"])
    index = faiss.read_index(paths["faiss_index"])
    
    if not isinstance(query, str):
        logger.error("Query must be a string.")
        raise ValueError("Query must be a string.")
    
    query_embedding = generate_embedding(query,embedding_model)
    
    query_embedding = query_embedding.reshape(1, -1)  # Convert to 2D array (1, embedding_dim)
    
    if query_embedding.shape[1] != embedding_dim:
        logger.error(f"Embedding dimension mismatch: got {query_embedding.shape[1]}, expected {embedding_dim}")
        raise ValueError(f"Embedding dim mismatch: got {query_embedding.shape[1]}, expected {embedding_dim}")  
    
    if index.ntotal == 0: 
        logger.warning("FAISS index is empty.")
        return [], [], []
    
    # Security Check for k not exceeding number of embedding 
    k = min(k, index.ntotal)
    
    distances, indices = index.search(query_embedding, k) 
    
    logger.info(f"Search completed. Found {len(indices[0])} results.")

    with open(paths["index_to_metadata"], "r", encoding="utf-8") as f:
        index_to_metadata = json.load(f)

    # Retrieve original chunks
    retrieved_texts = [index_to_metadata.get(str(idx), "[Not found]") for idx in indices[0]]
    logger.info(f"Retrieved {len(retrieved_texts)} texts for the query.")

    # logger.info("Top results:")
    # for i, text in enumerate(retrieved_texts):
    #     logger.info(f"{i+1}: {text[:100]}...")
    return indices[0], distances[0], retrieved_texts
 

def vector_db_pipeline(): 
    """
    End-to-end pipeline to generate and save embeddings, metadata and FAISS index.
    """
    generate_embeddings(paths["extracted_data"], paths["embedding_nomic"], paths["embedding_save"])
    generate_faiss_index()
 

# if __name__ == "__main__":
#     try:
#         vector_db_pipeline()
#         embedding_model =get_embedding_model()
#         logger.info(f"Using embedding model: {embedding_model}")
#         indices, distances , sentances= index_search("Spirit of America", k=5,embedding_model=embedding_model)
#         logger.info(f"Indices: {indices}")
#         logger.info(f"Distances: {distances}")
#     except Exception as e:
#         logger.error(f"Error generating FAISS index: {e}")
        
        
        