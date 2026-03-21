"""
MCP Tools for External Research Systems (Part IV, Point 11)

Three tools using Semantic Scholar API (free, no API key needed):

Tool 1 — Paper Metadata Lookup
    Input:  paper title
    Output: year, venue, citation count, authors, abstract, URL

Tool 2 — Related Work Discovery
    Input:  paper title
    Output: list of semantically related papers

Tool 3 — Trend Analytics
    Input:  keyword/topic
    Output: publication frequency over time + top papers on topic
"""

import requests
import time
from typing import List, Dict, Optional
from dataclasses import dataclass


# Base URL — no API key needed for basic usage
BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Fields to request for paper data
PAPER_FIELDS = "title,authors,year,venue,citationCount,abstract,url,externalIds"


@dataclass
class PaperMetadata:
    """Result from Tool 1 — Paper Metadata Lookup"""
    title: str
    authors: List[str]
    year: Optional[int]
    venue: str
    citation_count: int
    abstract: str
    url: str
    doi: str


@dataclass
class RelatedPaper:
    """Result from Tool 2 — Related Work Discovery"""
    title: str
    authors: List[str]
    year: Optional[int]
    venue: str
    citation_count: int
    url: str
    abstract: str


@dataclass
class TrendData:
    """Result from Tool 3 — Trend Analytics"""
    keyword: str
    papers_by_year: Dict[int, int]   # year → paper count
    top_papers: List[Dict]           # most cited papers on topic
    total_found: int


class SemanticScholarTools:
    """
    MCP Tools wrapper for Semantic Scholar API.

    All 3 tools are in this one class.
    Used by Dashboard, Library Chat, and Trends tabs.

    No API key needed — free tier allows 100 requests per 5 minutes.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ResearchPaperAssistant/1.0"
        })

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        """
        Make GET request to Semantic Scholar API.
        Returns JSON response or None on failure.

        Handles:
        - 429 rate limit: waits 10 seconds and retries once
        - 404 not found: returns None cleanly
        - Network errors: returns None cleanly
        """
        try:
            url = f"{BASE_URL}/{endpoint}"
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited — wait longer and retry once
                time.sleep(10)
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
                return None
            elif response.status_code == 404:
                return None
            else:
                return None

        except Exception:
            return None

    def _clean_title(self, title: str) -> str:
        """
        Clean title before searching Semantic Scholar.

        PDF extraction often adds noise:
        - "Attention Is All You Need 1"  (page number)
        - "ATTENTION IS ALL YOU NEED"    (all caps)
        - "Attention Is All You Need\n"  (newline)

        Returns cleaned title for better API match.
        """
        import re
        # Remove trailing numbers (page numbers)
        title = re.sub(r'\s+\d+\s*$', '', title)
        # Remove newlines and extra spaces
        title = re.sub(r'\s+', ' ', title).strip()
        # Remove special characters except hyphens and colons
        title = re.sub(r'[^\w\s\-:,]', '', title)
        return title

    # ----------------------------------------
    # TOOL 1 — Paper Metadata Lookup
    # Used in: Dashboard → Paper Viewer
    # Input: paper title string
    # Output: PaperMetadata with venue, citations, DOI etc.
    # ----------------------------------------

    def lookup_paper_metadata(self, title: str) -> Optional[PaperMetadata]:
        """
        Tool 1: Look up metadata for a paper by title.

        Real example:
        Input:  "Attention Is All You Need"
        Output: PaperMetadata(
            year=2017, venue="NeurIPS",
            citation_count=95000, doi="10.48550/arXiv.1706.03762"
        )

        Args:
            title: paper title to search for

        Returns:
            PaperMetadata or None if not found
        """

        # Clean title before searching
        title = self._clean_title(title)

        # Search by title
        params = {
            "query": title,
            "fields": PAPER_FIELDS,
            "limit": 1      # we only want the top match
        }

        data = self._get("paper/search", params)

        if not data or not data.get("data"):
            return None

        paper = data["data"][0]

        # Extract authors
        authors = [
            a.get("name", "") for a in paper.get("authors", [])
        ][:5]

        # Extract DOI
        doi = ""
        external_ids = paper.get("externalIds", {})
        if external_ids:
            doi = external_ids.get("DOI", "") or external_ids.get("ArXiv", "")

        return PaperMetadata(
            title=paper.get("title", title),
            authors=authors,
            year=paper.get("year"),
            venue=paper.get("venue", "") or "",
            citation_count=paper.get("citationCount", 0) or 0,
            abstract=paper.get("abstract", "") or "",
            url=paper.get("url", "") or "",
            doi=doi
        )

    # ----------------------------------------
    # TOOL 2 — Related Work Discovery
    # Used in: Library Chat → after any answer
    # Input: paper title
    # Output: list of related papers
    # ----------------------------------------

    def find_related_papers(self, title: str, limit: int = 5) -> List[RelatedPaper]:
        """
        Tool 2: Find papers related to a given paper.

        Steps:
        1. Search for the paper to get its Semantic Scholar ID
        2. Use the recommendations endpoint to get related papers

        Real example:
        Input:  "Attention Is All You Need"
        Output: [BERT, GPT-2, RoBERTa, XLNet, T5]

        Args:
            title: paper title to find related work for
            limit: number of related papers to return

        Returns:
            List of RelatedPaper objects
        """

        # Step 1: get paper ID
        search_params = {
            "query": title,
            "fields": "paperId,title",
            "limit": 1
        }

        search_data = self._get("paper/search", search_params)

        if not search_data or not search_data.get("data"):
            return []

        paper_id = search_data["data"][0].get("paperId")
        if not paper_id:
            return []

        # Step 2: get recommendations
        rec_params = {
            "fields": PAPER_FIELDS,
            "limit": limit
        }

        rec_data = self._get(
            f"recommendations/v1/papers/forpaper/{paper_id}",
            rec_params
        )

        # Recommendations endpoint uses different base URL
        # Fall back to search-based related papers if recommendations fail
        if not rec_data:
            return self._search_related_by_keywords(title, limit)

        papers = rec_data.get("recommendedPapers", [])

        results = []
        for p in papers[:limit]:
            authors = [a.get("name", "") for a in p.get("authors", [])][:3]
            results.append(RelatedPaper(
                title=p.get("title", ""),
                authors=authors,
                year=p.get("year"),
                venue=p.get("venue", "") or "",
                citation_count=p.get("citationCount", 0) or 0,
                url=p.get("url", "") or "",
                abstract=p.get("abstract", "") or ""
            ))

        return results

    def _search_related_by_keywords(self, title: str, limit: int) -> List[RelatedPaper]:
        """
        Fallback for Tool 2 — search by title keywords
        when recommendations endpoint fails.
        """

        # Use key words from title as search query
        keywords = " ".join(title.split()[:5])

        params = {
            "query": keywords,
            "fields": PAPER_FIELDS,
            "limit": limit + 1   # +1 because we skip the paper itself
        }

        data = self._get("paper/search", params)

        if not data or not data.get("data"):
            return []

        results = []
        title_lower = title.lower()

        for p in data.get("data", []):
            # Skip the paper itself
            if p.get("title", "").lower() == title_lower:
                continue

            authors = [a.get("name", "") for a in p.get("authors", [])][:3]
            results.append(RelatedPaper(
                title=p.get("title", ""),
                authors=authors,
                year=p.get("year"),
                venue=p.get("venue", "") or "",
                citation_count=p.get("citationCount", 0) or 0,
                url=p.get("url", "") or "",
                abstract=p.get("abstract", "") or ""
            ))

            if len(results) >= limit:
                break

        return results

    # ----------------------------------------
    # TOOL 3 — Trend Analytics
    # Used in: Trends Tab → Emerging Topics section
    # Input: keyword/topic
    # Output: paper counts per year + top papers
    # ----------------------------------------

    def get_trend_analytics(self, keyword: str, years: int = 8) -> Optional[TrendData]:
        """
        Tool 3: Get publication trend for a keyword/topic.

        Searches Semantic Scholar for papers on the topic,
        groups results by year to show publication frequency trend.

        Real example:
        Input:  "transformer attention"
        Output: TrendData(
            papers_by_year={2017: 45, 2018: 230, 2019: 890, ...},
            top_papers=[{title: "BERT", citations: 80000}, ...]
        )

        Args:
            keyword: topic to analyze
            years: how many recent years to show

        Returns:
            TrendData or None on failure
        """

        params = {
            "query": keyword,
            "fields": "title,year,citationCount,authors,venue",
            "limit": 100    # get 100 papers to analyze trend
        }

        data = self._get("paper/search", params)

        if not data or not data.get("data"):
            return None

        papers = data.get("data", [])
        total = data.get("total", len(papers))

        # Count papers by year
        year_counts: Dict[int, int] = {}
        for p in papers:
            year = p.get("year")
            if year and isinstance(year, int):
                year_counts[year] = year_counts.get(year, 0) + 1

        # Sort by citation count for top papers
        top_papers = sorted(
            [p for p in papers if p.get("citationCount")],
            key=lambda x: x.get("citationCount", 0),
            reverse=True
        )[:5]

        top_papers_clean = []
        for p in top_papers:
            authors = [a.get("name", "") for a in p.get("authors", [])][:2]
            top_papers_clean.append({
                "title": p.get("title", ""),
                "year": p.get("year"),
                "citation_count": p.get("citationCount", 0),
                "authors": authors,
                "venue": p.get("venue", "") or ""
            })

        return TrendData(
            keyword=keyword,
            papers_by_year=year_counts,
            top_papers=top_papers_clean,
            total_found=total
        )