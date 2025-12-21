from typing import List, Dict

import faiss
import numpy as np


class VectorDatabase:

    def __init__(self, dimension: int) -> None:

        # 3. Build FAISS index (FlatIP or FlatL2 depending on your choice)
        # dim = vectors.shape[1]
        self.dimension = dimension
        # self.index = faiss.IndexFlatIP(dimension)  # or IndexFlatL2(dim)    --> For testing

        self.index = faiss.IndexHNSWFlat(dimension, 32)   # M = number of neighbors per node in the HNSW graph (higher = better recall, more memory)
        # tune search speed/accuracy tradeoff
        self.index.hnsw.efSearch = 64                           # efSearch = number of nodes to explore during query (higher = better accuracy, slower)
        self.index.hnsw.efConstruction = 200                    # efConstruction = number of neighbors considered during index building (higher = better index quality, slower to build)

    def add(self, embeddings_array: np.ndarray):
        if embeddings_array.shape[1] != self.dimension:
            raise ValueError(f"Dimension mismatch: {embeddings_array.shape[0]} != {self.dimension}")
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)

    def save_index(self, path: str):
        # faiss.write_index(self.index, f"../data/faiss/recipe_ingredients_hnsw_{model_name}.index")
        pass

    def search(self, query_vector: np.ndarray, top_k: int = 10, top_p: float = 0.5) -> List[Dict]:
        faiss.normalize_L2(query_vector)
        distances, top_indices = self.index.search(query_vector, k=20)
        top_indices = top_indices[0]
        print(top_indices)
        d_min, d_max = distances.min(), distances.max()
        # normalized_distances = (distances - d_min) / (d_max - d_min)
        match_scores = 100 * (1 - normalized_distances)[0]
        return [
            {"score": score, "chunk_id": _id} for score, _id in zip(match_scores, top_indices)
        ]