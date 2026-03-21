"""
metadata_extractor.py — Extracts structured metadata from research paper PDFs.

Uses regex for most fields (fast, no API cost).
Uses LLM only for author extraction — regex fails on academic paper formats
due to superscript symbols, multi-column layouts, and affiliation text.

Called once per paper on first load; result cached in paper_registry.
"""

import re
import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader

from core.schema import ResearchPaper, PaperSection, CitationRelationship


class MetadataExtractor:
    """
    Extracts title, authors, abstract, year, keywords, sections,
    and citations from a research paper PDF.
    """

    def _load_pages(self, file_path: str, n_first: int = 3, n_last: int = 3):
        """
        Load first and last N pages from a PDF.

        Most metadata (title, authors, abstract) is on the first pages.
        References are on the last pages.

        Args:
            file_path: Path to the PDF.
            n_first: Number of pages to load from the start.
            n_last: Number of pages to load from the end.

        Returns:
            Tuple of (first_pages, last_pages, all_pages).
            Returns empty lists on failure.
        """
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            return pages[:n_first], pages[-n_last:] if len(pages) > n_first else [], pages
        except Exception:
            return [], [], []

    def _pages_to_text(self, pages) -> str:
        """Join a list of page Documents into one continuous text string."""
        return " ".join([p.page_content for p in pages])

    def _extract_title(self, text: str) -> str:
        """
        Extract paper title from first-page text.

        Heuristic: first line with 4-20 words that doesn't start with 'http'.

        Args:
            text: Text from the first few pages.

        Returns:
            Title string, or "Unknown Title" if not found.
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in lines[:10]:
            words = line.split()
            if 4 <= len(words) <= 20 and not line.startswith("http"):
                return line
        return "Unknown Title"

    def _extract_authors_with_llm(self, text: str, llm) -> List[str]:
        """
        Use LLM to extract author names from the first 600 chars of the paper.

        Regex-based extraction fails on academic PDFs due to superscript symbols
        (∗, †), multi-column text, and affiliation lines mixed with names.
        LLM handles all these formats reliably in one call.

        Args:
            text: First-page text of the paper.
            llm: LangChain LLM instance.

        Returns:
            List of author name strings (e.g. ["Ashish Vaswani", "Noam Shazeer"]).
            Returns empty list on failure.
        """
        first_600 = text[:600]

        prompt = f"""Extract only the personal author names from this academic paper text.

Rules:
- Return ONLY person names (First Last format)
- Separate names with commas
- Do NOT include university names, company names, or affiliations
- Do NOT include emails or symbols
- If you cannot find author names, return "Unknown"

Text:
{first_600}

Author names (comma separated):"""

        try:
            result = llm.invoke(prompt).content.strip().replace("\n", ",")
            names = [n.strip() for n in result.split(",") if n.strip()]

            # Filter out affiliations that slipped through
            affiliation_keywords = {
                "university", "institute", "lab", "research", "google",
                "facebook", "meta", "microsoft", "openai", "deepmind",
                "department", "school", "college", "brain", "ai", "nlp"
            }

            clean_names = []
            for name in names:
                if any(kw in name.lower() for kw in affiliation_keywords):
                    continue
                if len(name) < 4 or len(name) > 40:
                    continue
                if "unknown" in name.lower():
                    continue
                if len(name.split()) < 2:  # must have at least first + last name
                    continue
                clean_names.append(name)

            return clean_names[:10] if clean_names else []

        except Exception:
            return []

    def _extract_abstract(self, text: str) -> str:
        """
        Extract the abstract section from paper text.

        Finds text between the word "abstract" and the start of "introduction".

        Args:
            text: Text from the first few pages.

        Returns:
            Abstract text (max 1500 chars), or empty string if not found.
        """
        text_lower = text.lower()
        abstract_start = text_lower.find("abstract")
        if abstract_start == -1:
            return ""

        intro_start = text_lower.find("introduction", abstract_start + 1)
        end = intro_start if intro_start != -1 else abstract_start + 2000

        abstract = text[abstract_start + len("abstract"):end].strip()
        return re.sub(r"\s+", " ", abstract)[:1500]

    def _extract_year(self, text: str) -> Optional[int]:
        """
        Find the most frequently occurring 4-digit year (20xx) in the text.

        Most common year is usually the publication year.

        Args:
            text: First-page text.

        Returns:
            Year as int, or None if no year found.
        """
        years = re.findall(r"\b(20[0-2]\d)\b", text)
        if years:
            from collections import Counter
            return int(Counter(years).most_common(1)[0][0])
        return None

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from the paper.

        First tries to find an explicit "Keywords:" section.
        Falls back to extracting meaningful words from the title.

        Args:
            text: First-page text.

        Returns:
            List of keyword strings (max 10).
        """
        match = re.search(r"keywords?[:\s]+(.+?)[\n\.]", text, re.IGNORECASE)
        if match:
            return [k.strip() for k in re.split(r"[,;]", match.group(1)) if k.strip()][:10]

        # Fallback: use non-stopword title words as keywords
        title = self._extract_title(text)
        stopwords = {"a", "an", "the", "of", "in", "for", "on", "with", "and", "is", "are"}
        return [w.lower() for w in title.split()
                if w.lower() not in stopwords and len(w) > 3][:5]

    def _extract_citations(self, file_path: str, paper_id: str) -> List[CitationRelationship]:
        """
        Parse the reference section to extract structured citation data.

        Only handles numbered references ([1], [2], ... or 1., 2., ...).
        Author-year style references (Harvard format) are not supported.

        Args:
            file_path: Path to the PDF.
            paper_id: Filename used as the citing paper ID.

        Returns:
            List of CitationRelationship objects (max 50).
        """
        citations = []
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            last_pages_text = " ".join([p.page_content for p in pages[-4:]])

            ref_start = last_pages_text.lower().rfind("references")
            if ref_start == -1:
                return []

            ref_text = last_pages_text[ref_start:]

            for line in ref_text.split("\n"):
                line = line.strip()
                if not re.match(r"^\[\d+\]|\d+\.", line):
                    continue

                line_clean = line[:300]
                year_match = re.search(r"\b(19|20)\d{2}\b", line_clean)
                year = int(year_match.group()) if year_match else None

                title_match = re.search(r"\.\s+([A-Z][^.]+)\.", line_clean)
                title = title_match.group(1).strip() if title_match else line_clean[:100]

                citations.append(CitationRelationship(
                    citing_paper_id=paper_id,
                    cited_title=title,
                    cited_authors=[],
                    cited_year=year,
                    context=""
                ))

        except Exception:
            pass

        return citations[:50]

    def _extract_sections(self, all_pages) -> List[PaperSection]:
        """
        Detect and extract named sections from the full paper text.

        Uses the same regex boundary patterns as ingestion.py but creates
        PaperSection Pydantic objects instead of LangChain Documents.

        Args:
            all_pages: All page Documents from PyPDFLoader.

        Returns:
            List of PaperSection objects for each detected section.
        """
        section_patterns = [
            (r'\bAbstract\b(?=\s+[A-Z])', "abstract"),
            (r'\b1\s+Introduction\b', "introduction"),
            (r'\b2\s+(?:Related Work|Background)\b', "related_work"),
            (r'\b[23]\s+(?:Model Architecture|The Model|Methodology|Method|Approach)\b', "method"),
            (r'\b[456]\s+(?:Experiment|Result|Evaluation|Training)\b', "experiments"),
            (r'\b[6789]\s+Conclusion\b', "conclusion"),
        ]

        full_text = " ".join([p.page_content for p in all_pages])

        # Build page offset map for page number tracking
        page_map = []
        offset = 0
        for p in all_pages:
            page_map.append((offset, p.metadata.get("page", 0)))
            offset += len(p.page_content) + 1

        # Find first match for each pattern (avoid duplicates)
        boundaries = []
        for pattern, name in section_patterns:
            m = re.search(pattern, full_text)
            if m and name not in [n for _, n in boundaries]:
                boundaries.append((m.start(), name))

        boundaries.sort(key=lambda x: x[0])

        sections = []
        for i, (pos, name) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            text = re.sub(r"\s+", " ", full_text[pos:end]).strip()[:3000]

            page_num = 0
            for off, pnum in page_map:
                if off <= pos:
                    page_num = pnum

            if text:
                sections.append(PaperSection(name=name, content=text, page=page_num))

        return sections

    def extract(self, file_path: str, paper_id: str,
                chunk_count: int = 0, llm=None) -> ResearchPaper:
        """
        Run the full metadata extraction pipeline for one paper.

        Extracts all fields required by the ResearchPaper Pydantic model.
        Authors are extracted via LLM if an llm instance is provided;
        otherwise the authors field is left empty.

        Args:
            file_path: Path to the PDF file.
            paper_id: Filename used as the unique paper identifier.
            chunk_count: Number of FAISS chunks created during ingestion.
            llm: Optional LangChain LLM for author extraction.
                 If None, authors will be empty until enriched via Semantic Scholar.

        Returns:
            Fully populated ResearchPaper object.
        """
        first_pages, last_pages, all_pages = self._load_pages(file_path)
        first_text = self._pages_to_text(first_pages)
        full_text = self._pages_to_text(all_pages)
        file_size_kb = round(os.path.getsize(file_path) / 1024, 1)

        # Author extraction: LLM if available, else empty
        authors = self._extract_authors_with_llm(first_text, llm) if llm else []

        # Extract raw reference strings from last pages
        raw_refs = []
        if all_pages:
            last_text = self._pages_to_text(last_pages)
            ref_start = last_text.lower().rfind("references")
            if ref_start != -1:
                for line in last_text[ref_start:].split("\n"):
                    line = line.strip()
                    if re.match(r"^\[\d+\]|\d+\.", line):
                        raw_refs.append(line[:200])

        return ResearchPaper(
            paper_id=paper_id,
            title=self._extract_title(first_text),
            authors=authors,
            abstract=self._extract_abstract(first_text),
            full_text=full_text[:5000],
            year=self._extract_year(first_text),
            venue="",
            keywords=self._extract_keywords(first_text),
            references=raw_refs[:30],
            citations=self._extract_citations(file_path, paper_id),
            sections=self._extract_sections(all_pages),
            chunk_count=chunk_count,
            file_size_kb=file_size_kb
        )