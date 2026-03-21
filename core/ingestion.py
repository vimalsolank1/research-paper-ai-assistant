"""
ingestion.py — PDF loading and section-based chunking.

Instead of splitting by character count, we detect actual section headings
(Abstract, Introduction, Method, etc.) and create one chunk per section.
This gives much better retrieval accuracy for academic papers.
"""

from typing import List, Tuple
from pathlib import Path
import os
import re

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


# Regex patterns to detect section headings in continuous PDF text.
# PyPDFLoader outputs text without clean newlines, so we use lookaheads
# instead of line-start anchors.
# Format: (pattern, section_label)
SECTION_BOUNDARIES = [
    (r'\bAbstract\b(?=\s+[A-Z])', "abstract"),
    (r'\b1\s+Introduction\b', "introduction"),
    (r'\bIntroduction\b(?=\s+[A-Z])', "introduction"),
    (r'\b2\s+(?:Related Work|Background)\b', "related_work"),
    (r'\b(?:Related Work|Background)\b(?=\s+[A-Z])', "related_work"),
    (r'\b[23]\s+(?:Model Architecture|The Model|Methodology|Method|Approach|Proposed)\b', "method"),
    (r'\bModel Architecture\b(?=\s+[A-Z])', "method"),
    (r'\b[456]\s+(?:Experiment|Result|Evaluation|Training)\b', "experiments"),
    (r'\b(?:Experiments?|Results?|Evaluation)\b(?=\s+[A-Z0-9])', "experiments"),
    (r'\b[6789]\s+Conclusion\b', "conclusion"),
    (r'\bConclusion\b(?=\s+[A-Z])', "conclusion"),
    (r'\bReferences\b(?=\s+\[)', "references"),
]


class DocumentProcessor:
    """
    Loads PDFs and splits them into section-based chunks for FAISS indexing.

    Primary method: process(file_path) → List[Document]
    Each returned Document has metadata: paper_id, source, section, page, chunk_id
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Args:
            chunk_size: Max chars per chunk (fallback splitter only).
            chunk_overlap: Overlap between fallback chunks.
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE or 1500
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP or 300

        # Used only when a section is too long or heading detection fails
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "]
        )

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a PDF or TXT file into page-level Documents.

        Args:
            file_path: Path to the file.

        Returns:
            List of Documents, one per page.

        Raises:
            ValueError: If file type is not .pdf or .txt.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return loader.load()

    def _clean_text(self, text: str) -> str:
        """Collapse all whitespace to single spaces."""
        return re.sub(r'\s+', ' ', text).strip()

    def _add_metadata(self, documents: List[Document], file_path: str) -> List[Document]:
        """
        Attach paper_id, source, and source_type to every page document.

        Args:
            documents: Page-level documents from load_document().
            file_path: Original file path (used to extract filename).

        Returns:
            Same documents with updated metadata.
        """
        file_name = os.path.basename(file_path)
        for doc in documents:
            doc.metadata.update({
                "paper_id": file_name,
                "source": file_name,
                "source_type": Path(file_path).suffix.replace(".", "")
            })
        return documents

    def _find_sections(self, full_text: str) -> List[Tuple[int, str]]:
        """
        Find character positions of section headings in the full text.

        Args:
            full_text: All pages joined into one string.

        Returns:
            List of (char_position, section_name) sorted by position.
        """
        found = []

        for pattern, section_name in SECTION_BOUNDARIES:
            for match in re.finditer(pattern, full_text):
                pos = match.start()
                # Keep only the first occurrence of each section label
                if section_name not in [s for _, s in found]:
                    found.append((pos, section_name))

        found.sort(key=lambda x: x[0])
        return found

    def _split_by_sections(
        self,
        full_text: str,
        boundaries: List[Tuple[int, str]],
        page_map: List[Tuple[int, int]]
    ) -> List[Tuple[str, str, int]]:
        """
        Cut the full text at section boundaries.

        Args:
            full_text: Complete joined PDF text.
            boundaries: Section positions from _find_sections().
            page_map: List of (char_offset, page_number) for page tracking.

        Returns:
            List of (section_text, section_name, page_number).
        """
        if not boundaries:
            return [(full_text, "unknown", 0)]

        results = []

        for i, (pos, section_name) in enumerate(boundaries):
            # Section ends where the next one begins
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            section_text = self._clean_text(full_text[pos:end])

            # Find which page this section starts on
            page_num = 0
            for offset, pnum in page_map:
                if offset <= pos:
                    page_num = pnum

            if section_text:
                results.append((section_text, section_name, page_num))

        # Capture title/author text before the first section heading
        if boundaries and boundaries[0][0] > 100:
            pre_text = self._clean_text(full_text[:boundaries[0][0]])
            if pre_text:
                results.insert(0, (pre_text, "title", 0))

        return results

    def process(self, file_path: str) -> List[Document]:
        """
        Full pipeline: PDF → section-based LangChain Document chunks.

        Steps:
            1. Load PDF pages
            2. Join pages into one text, track page offsets
            3. Detect section boundaries via regex
            4. Split at boundaries; fallback to page-position if < 2 sections found
            5. Skip references and title sections
            6. Split oversized sections with fallback splitter

        Args:
            file_path: Path to PDF or TXT file.

        Returns:
            List of Documents with metadata:
            paper_id, source, section, page, chunk_id
        """
        pages = self.load_document(file_path)
        pages = self._add_metadata(pages, file_path)
        file_name = os.path.basename(file_path)

        # Join all pages into one string and track page start positions
        full_text_parts = []
        page_map = []
        current_offset = 0

        for page_doc in pages:
            page_num = page_doc.metadata.get("page", 0)
            text = page_doc.page_content
            page_map.append((current_offset, page_num))
            full_text_parts.append(text)
            current_offset += len(text) + 1  # +1 for the space separator

        full_text = " ".join(full_text_parts)

        # Try section-based splitting; fall back if too few headings found
        boundaries = self._find_sections(full_text)
        if len(boundaries) >= 2:
            sections = self._split_by_sections(full_text, boundaries, page_map)
        else:
            sections = self._fallback_page_sections(pages)

        chunks = []
        chunk_id = 0
        skip_sections = {"references", "title"}

        for section_text, section_name, page_num in sections:

            if section_name in skip_sections or not section_text.strip():
                continue

            if len(section_text) <= self.chunk_size * 2:
                # Section fits in one chunk — keep it whole for full context
                doc = Document(
                    page_content=section_text,
                    metadata={
                        "paper_id": file_name,
                        "source": file_name,
                        "section": section_name,
                        "page": page_num,
                        "chunk_id": chunk_id,
                    }
                )
                chunks.append(doc)
                chunk_id += 1
            else:
                # Very long section — split into overlapping sub-chunks
                sub_docs = self.fallback_splitter.create_documents(
                    [section_text],
                    metadatas=[{
                        "paper_id": file_name,
                        "source": file_name,
                        "section": section_name,
                        "page": page_num,
                    }]
                )
                for sub_doc in sub_docs:
                    sub_doc.metadata["chunk_id"] = chunk_id
                    chunks.append(sub_doc)
                    chunk_id += 1

        return chunks

    def _fallback_page_sections(self, pages: List[Document]) -> List[Tuple[str, str, int]]:
        """
        Assign section labels by page position when heading detection fails.

        Heuristic: first page = abstract, last 2 pages = conclusion, middle = method.

        Args:
            pages: Page-level Documents from load_document().

        Returns:
            List of (text, section_label, page_number).
        """
        total = len(pages)
        results = []

        for i, page in enumerate(pages):
            text = self._clean_text(page.page_content)
            page_num = page.metadata.get("page", i)

            if i == 0:
                section = "abstract"
            elif i <= 2:
                section = "introduction"
            elif i >= total - 2:
                section = "conclusion"
            elif i <= total // 2:
                section = "method"
            else:
                section = "experiments"

            if text:
                results.append((text, section, page_num))

        return results

    def process_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Process a raw text string into chunks (no section detection).

        Useful for ingesting plain text content outside of a PDF.

        Args:
            text: Raw text to chunk.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            List of Documents with chunk_id and section='unknown'.
        """
        metadata = metadata or {}
        doc = Document(page_content=self._clean_text(text), metadata=metadata)
        sub_docs = self.fallback_splitter.split_documents([doc])

        for i, d in enumerate(sub_docs):
            d.metadata["chunk_id"] = i
            d.metadata["section"] = "unknown"

        return sub_docs