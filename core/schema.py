"""
schema.py — Pydantic data models for the research paper system.

Three models required by the project specification:
    1. PaperSection         — one section of a paper (Abstract, Method, etc.)
    2. CitationRelationship — a citation link between two papers
    3. ResearchPaper        — unified representation of a full paper

All models are stored in paper_registry (session state) after extraction
and used by chat.py, dashboard.py, and trends.py for display and retrieval.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PaperSection(BaseModel):
    """
    Represents one extracted section from a research paper.

    Created by MetadataExtractor._extract_sections() and stored
    in ResearchPaper.sections. Used by chat.py to answer section-specific
    questions (e.g. "what is the abstract?") without going through FAISS.

    Attributes:
        name: Section label — "abstract", "introduction", "method", etc.
        content: Full text of the section (max 3000 chars).
        page: Page number where the section starts in the PDF.
    """
    name: str
    content: str
    page: int = 0


class CitationRelationship(BaseModel):
    """
    Represents a citation from one paper to another.

    Extracted from the References section of a paper and stored in
    ResearchPaper.citations. Used by the Trends tab to build the citation network.

    Attributes:
        citing_paper_id: Filename of the paper containing this citation.
        cited_title: Title of the paper being cited.
        cited_authors: Authors of the cited paper (may be empty if not parsed).
        cited_year: Publication year of the cited paper.
        context: Sentence in which the citation appears (empty if not extracted).
    """
    citing_paper_id: str
    cited_title: str
    cited_authors: List[str] = Field(default_factory=list)
    cited_year: Optional[int] = None
    context: str = ""


class ResearchPaper(BaseModel):
    """
    Unified internal representation of a research paper.

    Populated by MetadataExtractor.extract() on first load and cached in
    st.session_state.paper_registry. All UI tabs read from this model.

    Attributes:
        paper_id: Unique identifier — the PDF filename.
        title: Paper title extracted from the first page.
        authors: List of author names (LLM-extracted).
        abstract: Abstract text (max 1500 chars).
        full_text: First 5000 chars of the full paper text.
        year: Publication year (regex-extracted, overridden by Semantic Scholar).
        venue: Conference or journal name (empty until Semantic Scholar enrichment).
        keywords: Topic keywords from the paper or derived from the title.
        references: Raw reference strings from the References section.
        citations: Structured CitationRelationship objects.
        sections: List of PaperSection objects for each detected section.
        chunk_count: Number of FAISS chunks created during ingestion.
        file_size_kb: PDF file size in kilobytes.
    """
    paper_id: str
    title: str = "Unknown Title"
    authors: List[str] = Field(default_factory=list)
    abstract: str = ""
    full_text: str = ""
    year: Optional[int] = None
    venue: str = ""
    keywords: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    citations: List[CitationRelationship] = Field(default_factory=list)
    sections: List[PaperSection] = Field(default_factory=list)
    chunk_count: int = 0
    file_size_kb: float = 0.0

    def get_section(self, name: str) -> Optional[PaperSection]:
        """
        Look up a section by name (case-insensitive).

        Args:
            name: Section label to find (e.g. "abstract", "conclusion").

        Returns:
            Matching PaperSection, or None if not found.
        """
        for sec in self.sections:
            if sec.name.lower() == name.lower():
                return sec
        return None

    def get_citation_count(self) -> int:
        """Return the number of structured citations extracted from this paper."""
        return len(self.citations)

    def get_cited_papers(self) -> List[str]:
        """Return a list of titles of all papers cited by this paper."""
        return [c.cited_title for c in self.citations]