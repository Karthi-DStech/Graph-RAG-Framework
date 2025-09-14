from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    Handles document ingestion and preprocessing.

    This class:
        1. Recursively scans an input directory for supported file types.
        2. Loads each file into LangChain `Document` objects.
        3. Cleans text (basic normalization).
        4. Splits large documents into smaller overlapping chunks suitable
           for embedding or graph extraction.

    Parameters
    ----------
    opt : argparse.Namespace or similar
        Parsed configuration object. Must include:
            - input_dir (str): Path to directory containing documents.
            - chunk_size (int): Character size of each chunk.
            - chunk_overlap (int): Overlap between chunks.
            - extensions (List[str], optional): Allowed file extensions.
    """

    def __init__(self, opt):
        self._opt = opt

    def _load_all_documents(self) -> List[Document]:
        input_dir = Path(self._opt.input_dir)
        exts = {e.lower() for e in getattr(self._opt, "extensions", [".pdf"])}

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        documents: List[Document] = []

        for path in input_dir.rglob("*"):
            if path.suffix.lower() in exts:

                if path.suffix.lower() == ".pdf":
                    documents.extend(self._load_pdf(path))

                elif path.suffix.lower() in {".txt", ".md"}:
                    documents.extend(self._load_text(path))

                # add other file types here as needed

                else:
                    raise ValueError(f"Unsupported file extension: {path.suffix}")

        return documents

    def _load_pdf(self, path: Path) -> List[Document]:
        """
        Recursively load all supported files from the input directory.

        Returns
        -------
        List[Document]
            A list of LangChain Document objects with basic metadata.

        Raises
        ------
        FileNotFoundError
            If the input directory does not exist.
        ValueError
            If an encountered file has an unsupported extension.
        """

        loader = PyPDFLoader(str(path))
        pages = loader.load_and_split()

        for p in pages:
            p.metadata = {**p.metadata, "source": str(path), "file_name": path.name}
        return pages

    def process_documents(self) -> List[Document]:
        """
        Load, clean, and split documents into chunks with metadata.

        Workflow:
            1. Load all documents from input_dir.
            2. Clean each document's text.
            3. Split into overlapping chunks.
            4. Add chunk-level metadata.

        Returns
        -------
        List[Document]
            Chunked documents ready for embedding or graph ingestion.
        """
        # 1. Load everything
        raw_docs = self._load_all_documents()

        # 2. Simple cleaning (extend as needed)
        cleaned_docs = []
        for d in raw_docs:
            text = self._clean_text(d.page_content)
            if text.strip():
                cleaned_docs.append(Document(page_content=text, metadata=d.metadata))

        # 3. Split into chunks
        chunk_size = self._opt.chunk_size
        chunk_overlap = self._opt.chunk_overlap

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunked_docs = splitter.split_documents(cleaned_docs)

        # 4. Add chunk index to metadata for traceability
        for idx, doc in enumerate(chunked_docs):
            doc.metadata = {
                **doc.metadata,
                "chunk_index": idx,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }

        return chunked_docs
