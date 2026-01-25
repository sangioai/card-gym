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
) -> Tuple[faiss.Index, np.ndarray]:
    index.add(embeddings)
    return index

# --- Search function ------------------------------------------------

def search(
    query: str|list,
    documents: List[str],
    encoder: SentenceTransformer,
    index: faiss.Index,
    k: int = 5,
    **args
) -> List[Tuple[str, float]]:
    """
    Returns top-k (document, score) pairs.
    """

    query_embedding = encoder.encode(
        query if isinstance(query, list) else [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        **args
    ).astype("float32")

    scores, indices = index.search(query_embedding, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append((documents[idx], float(score)))

    return results

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
    # load model
    model = load_text_embedder()
    # create index
    # Node: HNSW graph indexing does not properly work with faiss
    index, _ = build_index(documents, model, use_hnsw=True)
    # perform search
    results = search(
        query="I like eating pasta",
        documents=documents,
        encoder=model,
        index=index,
        k=3,
        prompt_name="query"
    )
    # score results
    for doc, score in results:
        print(f"{doc}  (score={score:.3f})")