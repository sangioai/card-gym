import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

DEFAULT_RETRIEVER_DIM = 384

##################################
# Encoder Functions
##################################

# --- Model loading ------------------------------------------------

def load_text_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)

# --- Model embedding ------------------------------------------------

def embed_state(model, states):
    embeddings = model.encode(
        states,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    return embeddings


##################################
# Index Functions
##################################

# --- Index building ------------------------------------------------

def build_index(
    dim = DEFAULT_RETRIEVER_DIM,
    use_hnsw: bool = False,
    quantized: bool = False,
) -> Tuple[faiss.Index, np.ndarray]:
    if use_hnsw:
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efSearch = 64
    elif quantized:
        index = faiss.IndexScalarQuantizer(
            dim,
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT
        )
    else:
        index = faiss.IndexFlatIP(dim)
        
    return index

# --- Insert function ------------------------------------------------

def add_embeddings(
    index: faiss.Index,
    embeddings
) -> faiss.Index:
    index.add(embeddings)
    return index

# --- Search function ------------------------------------------------

def search(
    query_embeddings: str|list,
    documents: List[str],
    index: faiss.Index,
    k: int = 5,
    return_idx = False
) -> List[Tuple[str, float]]:
    """
    Returns top-k (document, score) pairs.
    """

    scores, indices = index.search(query_embeddings, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append((documents[idx], float(score)))

    return results if not return_idx else (results, indices)

# --- Save/load functions ------------------------------------------------

def save_index(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

##################################
# Main
##################################

if __name__ == "__main__":
    # create documents
    documents = [
        "I love pizza and pasta",
        "The cat is sleeping on the couch",
        "Machine learning is fascinating",
        "I enjoy Italian food",
        "Dogs are great pets",
    ]
    # Load model
    model = load_text_embedder()
    # Embed documents
    doc_embeddings = embed_state(model, documents)
    # Build index
    index = build_index(use_hnsw=True)
    # Add embeddings to index  <-- THIS WAS MISSING
    add_embeddings(index, doc_embeddings)
    # Embed query
    query_embeddings = embed_state(model, ["I like eating pasta"])
    # perform search
    results = search(
        query_embeddings=query_embeddings,
        documents=documents,
        index=index,
        k=5
    )
    # score results
    for doc, score in results:
        print(f"{doc}  (score={score:.3f})")