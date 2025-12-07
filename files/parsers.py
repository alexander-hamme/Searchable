import os

import fitz


class FileParser:

    def __init__(self):
        self.pdf_db = PDFDatabase()

    def parse_file(self, filename):
        ext = os.path.splitext(filename)[1]
        match ext:
            case '.pdf':
                self.pdf_db.ingest_pdf(filename)
                return self.pdf_db.get_pdf_as_chunks(filename)

            case _: raise NotImplementedError(f"Unsupported file type: {ext}")


class PDFDatabase:

    def __init__(self):
        self.pdf_mapper = {}

    def ingest_pdf(self, filepath):

        with fitz.open(filepath) as doc:
            page_map = []
            text_chunks = []

            char_offset = 0
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()  # simple text extraction
                page_map.append((page_num, char_offset))
                text_chunks.append(text)
                char_offset += len(text)

        self.pdf_mapper[filepath] = {
            "page_map": page_map,
            "text_chunks": text_chunks
        }

    def get_pdf_as_chunks(self, filepath):
        return self.pdf_mapper[filepath]["text_chunks"]

    def offset_to_page(self, filepath, start_char):
        # page_map is arranged in-order by start_char
        # find last page whose start_char <= our start_char
        last_page = -1
        for page_num, page_start in self.pdf_mapper[filepath]["page_map"]:
            if page_start <= start_char:
                last_page = page_num
            else:
                break

        return last_page
