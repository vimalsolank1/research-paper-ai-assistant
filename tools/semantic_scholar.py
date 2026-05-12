"""
Semantic Scholar API tools used for:
- paper metadata lookup
- related paper discovery
- research trend analysis
"""

import requests
import time
from typing import List, Dict, Optional
from dataclasses import dataclass


# Base API URL
BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Fields fetched from API response
PAPER_FIELDS = "title,authors,year,venue,citationCount,abstract,url,externalIds"


@dataclass
class PaperMetadata:
    """
    Metadata returned from paper lookup.
    """
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
    """
    Structure for related paper results.
    """
    title: str
    authors: List[str]
    year: Optional[int]
    venue: str
    citation_count: int
    url: str
    abstract: str


@dataclass
class TrendData:
    """
    Structure for trend analytics results.
    """
    keyword: str
    papers_by_year: Dict[int, int]   # year -> paper count
    top_papers: List[Dict]           # top cited papers
    total_found: int


class SemanticScholarTools:
    """
    Wrapper around Semantic Scholar API tools.

    This class handles:
    - paper metadata lookup
    - related paper recommendations
    - research trend analysis

    Used across dashboard, chat, and trends modules.
    """

    def __init__(self):

        # Reusable request session
        self.session = requests.Session()

        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ResearchPaperAssistant/1.0"
        })

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        """
        Send GET request to Semantic Scholar API.

        Handles:
        - successful responses
        - retry on API rate limit
        - connection failures safely

        Args:
            endpoint: API endpoint path.
            params: Query parameters for request.

        Returns:
            Parsed JSON response or None on failure.
        """
        try:
            url = f"{BASE_URL}/{endpoint}"

            response = self.session.get(
                url,
                params=params,
                timeout=10
            )

            # Successful response
            if response.status_code == 200:
                return response.json()

            # Retry once if API rate limit is hit
            elif response.status_code == 429:

                time.sleep(10)

                response = self.session.get(
                    url,
                    params=params,
                    timeout=10
                )

                if response.status_code == 200:
                    return response.json()

                return None

            # Paper not found
            elif response.status_code == 404:
                return None

            else:
                return None

        except Exception:
            return None

    def _clean_title(self, title: str) -> str:
        """
        Clean extracted paper title before API search.

        PDF extraction sometimes adds:
        - page numbers
        - extra spaces
        - unwanted characters

        This improves title matching accuracy.
        """

        import re

        # Remove trailing page numbers
        title = re.sub(r'\s+\d+\s*$', '', title)

        # Remove extra spaces and newlines
        title = re.sub(r'\s+', ' ', title).strip()

        # Remove unwanted special characters
        title = re.sub(r'[^\w\s\-:,]', '', title)

        return title

    # ----------------------------------------
    # TOOL 1 — Paper Metadata Lookup
    # Used in Dashboard -> Paper Viewer
    # ----------------------------------------

    def lookup_paper_metadata(self, title: str) -> Optional[PaperMetadata]:
        """
        Fetch metadata for a research paper using its title.

        Args:
            title: Research paper title.

        Returns:
            PaperMetadata object with authors, venue,
            citations, DOI, abstract, and URL.
        """

        # Clean title before API search
        title = self._clean_title(title)

        # Search paper by title
        params = {
            "query": title,
            "fields": PAPER_FIELDS,
            "limit": 1
        }

        data = self._get("paper/search", params)

        if not data or not data.get("data"):
            return None

        paper = data["data"][0]

        # Extract top author names
        authors = [
            a.get("name", "") for a in paper.get("authors", [])
        ][:5]

        # Extract DOI or arXiv ID
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
    # Used in Library Chat
    # ----------------------------------------

    def find_related_papers(self, title: str, limit: int = 5) -> List[RelatedPaper]:
        """
        Find papers related to the given research paper.

        First searches for the paper ID, then fetches
        recommendations from Semantic Scholar.

        Args:
            title: Paper title to search.
            limit: Number of related papers to return.

        Returns:
            List of RelatedPaper objects.
        """

        # Search for Semantic Scholar paper ID
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

        # Fetch recommended papers
        rec_params = {
            "fields": PAPER_FIELDS,
            "limit": limit
        }

        rec_data = self._get(
            f"recommendations/v1/papers/forpaper/{paper_id}",
            rec_params
        )

        # Use fallback keyword search if recommendations fail
        if not rec_data:
            return self._search_related_by_keywords(title, limit)

        papers = rec_data.get("recommendedPapers", [])

        results = []

        for p in papers[:limit]:

            authors = [
                a.get("name", "") for a in p.get("authors", [])
            ][:3]

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
        Fallback related paper search using title keywords.

        Used when the recommendation API fails.

        Args:
            title: Original paper title.
            limit: Maximum number of papers to return.

        Returns:
            List of related papers from keyword search.
        """

        # Use first few title words as search keywords
        keywords = " ".join(title.split()[:5])

        params = {
            "query": keywords,
            "fields": PAPER_FIELDS,
            "limit": limit + 1
        }

        data = self._get("paper/search", params)

        if not data or not data.get("data"):
            return []

        results = []

        title_lower = title.lower()

        for p in data.get("data", []):

            # Skip original paper from results
            if p.get("title", "").lower() == title_lower:
                continue

            authors = [
                a.get("name", "") for a in p.get("authors", [])
            ][:3]

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
    # Used in Trends Tab
    # ----------------------------------------

    def get_trend_analytics(self, keyword: str, years: int = 8) -> Optional[TrendData]:
        """
        Analyze publication trends for a research topic.

        Fetches papers related to the keyword and
        groups them by publication year.

        Args:
            keyword: Topic or research area.
            years: Number of recent years to analyze.

        Returns:
            TrendData object with yearly counts
            and top cited papers.
        """

        # Fetch papers related to topic
        params = {
            "query": keyword,
            "fields": "title,year,citationCount,authors,venue",
            "limit": 100
        }

        data = self._get("paper/search", params)

        if not data or not data.get("data"):
            return None

        papers = data.get("data", [])
        total = data.get("total", len(papers))

        # Count papers published per year
        year_counts: Dict[int, int] = {}

        for p in papers:

            year = p.get("year")

            if year and isinstance(year, int):
                year_counts[year] = year_counts.get(year, 0) + 1

        # Sort papers by citation count
        top_papers = sorted(
            [p for p in papers if p.get("citationCount")],
            key=lambda x: x.get("citationCount", 0),
            reverse=True
        )[:5]

        top_papers_clean = []

        for p in top_papers:

            authors = [
                a.get("name", "") for a in p.get("authors", [])
            ][:2]

            # Keep only required fields for UI
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