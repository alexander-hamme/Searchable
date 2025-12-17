import asyncio
from typing import List, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi.concurrency import run_in_threadpool

from abc import ABC, abstractmethod
from typing import List

class EmbeddingClient(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        ...


class LocalEmbeddingClient(EmbeddingClient):

    BATCH_SIZE = 32

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(self.dimension)

    def _embed_sync(self, text_chunks: List[str]) -> np.ndarray:
        return self.model.encode(
            text_chunks,
            batch_size=self.BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        # return run_in_threadpool(self._embed_sync, texts)
        return self._embed_sync(texts)



class HttpEmbeddingClient(EmbeddingClient):
    def __init__(self, base_url: str):
        self.base_url = base_url
        # self.client = httpx.Client(timeout=30)

    def embed_documents(self, texts):
        resp = self.client.post(
            f"{self.base_url}/embed",
            json={"texts": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return np.array(data["embeddings"], dtype="float32")
