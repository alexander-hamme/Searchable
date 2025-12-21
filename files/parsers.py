import asyncio
import glob
import os

import faiss
import numpy as np
import pymupdf

from database.vector_db import VectorDatabase
from database.metadata_db import DocChunkDb, DocMetadata
from embed.embedders import LocalEmbeddingClient


class FileIngester:

    TEXT_CHUNK_SIZE = 300

    def __init__(self, embedding_client: LocalEmbeddingClient, vector_db: VectorDatabase):
        self.current_chunk_id = 0
        self.doc_chunk_db = DocChunkDb()
        self.embedding_client = embedding_client
        self.faiss_manager = vector_db

    def parse_all_files(self, directory: str):

        filepaths = [matches for ext in ["pdf", "txt"]
                     if len(matches := glob.glob(f"{directory}/*.{ext}")) > 0]

        print(filepaths)
        filepaths = [path for lst in filepaths for path in lst]

        print(filepaths)

        for path in filepaths:
            vector_array = self.ingest_single_file(path)
            if vector_array is None:
                continue
            self.faiss_manager.add(vector_array)

        # # 3. Build FAISS index (FlatIP or FlatL2 depending on your choice)
        # dim = self.faiss_manager.dimension
        # vectors.shape[1]
        # #TODO  index = faiss.IndexFlatIP(dim)  # or IndexFlatL2(dim)
        #
        # # NOTE: `parse_all_files` must be `async def` for this to work
        #
        # for vector_array in await asyncio.gather(*tasks):
            '''tasks = [self.ingest_file(path) for path in filepaths]


        # 3. Build FAISS index (FlatIP or FlatL2 depending on your choice)
        dim = self.faiss_manager.dimension
        vectors.shape[1]
        #TODO  index = faiss.IndexFlatIP(dim)  # or IndexFlatL2(dim)

        # NOTE: `parse_all_files` must be `async def` for this to work
        
        for vector_array in await asyncio.gather(*tasks):
            self.faiss_manager.add(vector_array)'''


    def ingest_single_file(self, filepath):
        ext = os.path.splitext(filepath)[1]
        match ext:
            case '.pdf':
                return self._ingest_and_vectorize(iter(self._iter_pdf_pages(filepath)))

            case _:
                print(f"Unsupported file type: {ext}  ({filepath})")
                return None

        #raise NotImplementedError(f"Unsupported file type: {ext}")

    def retrieve_chunk_by_id(self, chunk_id: int) -> np.ndarray:
        return self.doc_chunk_db.get(chunk_id)

    def _chunk_text(self, page_text: str):
        # TODO split by words / phrases     --> need to check length and use threshold
        for text_idx in range(0, len(page_text), self.TEXT_CHUNK_SIZE):
            yield self.current_chunk_id, page_text[text_idx:text_idx + self.TEXT_CHUNK_SIZE]
            self.current_chunk_id += 1

    def _ingest_and_vectorize(self, page_iterator) -> np.ndarray:
        """
        Processes and vectorizes text data using a page iterator. The method retrieves
        text chunks from the provided pages, processes them, and stores metadata
        information for each chunk. This data is then sent to an embedding client
        to generate corresponding vector embeddings.

        :param page_iterator: An iterator providing tuples of (page_no, page_text),
            where page_no is the page number and page_text is the textual content
            of that page.
        :return: A NumPy array containing vectorized embeddings of the processed
            text chunks.
        :rtype: np.ndarray
        """
        text_chunks = []
        for page_no, page_text in page_iterator:
            # doc_metadata = DocMetadata(page_no, None)
            for chunk_id, text_chunk in self._chunk_text(page_text):
                self.doc_chunk_db.insert_chunk(chunk_id, {'page_no': page_no}, text_chunk)  # Paragraph number?
                text_chunks.append(text_chunk)

        return self.embedding_client.embed_documents(text_chunks)

    @staticmethod
    def _iter_pdf_pages(filepath):
        with pymupdf.open(filepath) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()  # simple text extraction
                yield page_num, text





    # def offset_to_page(self, filepath, start_char):
    #     # page_map is arranged in-order by start_char
    #     # find last page whose start_char <= our start_char
    #     last_page = -1
    #     for page_num, page_start in self.pdf_mapper[filepath]["page_map"]:
    #         if page_start <= start_char:
    #             last_page = page_num
    #         else:
    #             break
    #
    #     return last_page


    # fp.parse_all_files("/Users/alex/Documents/Searchable/SampleData")
