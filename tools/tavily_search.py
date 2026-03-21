from typing import List
from langchain_core.documents import Document
from tavily import TavilyClient

from config.settings import settings


class TavilySearchTool:
    """
    Handles web search using Tavily API.

    Converts web results into Document format for RAG usage.
    """

    def __init__(self):
        """
        Initialize Tavily client.
        """
        self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)

    def search(self, query: str, max_results: int = 5) -> List[Document]:
        """
        Perform web search.

        Args:
            query (str): Search query
            max_results (int): Number of results

        Returns:
            List[Document]: Web results as documents
        """

        response = self.client.search(
            query=query,
            max_results=max_results
        )

        documents = []

        for result in response.get("results", []):
            content = result.get("content", "")
            url = result.get("url", "unknown")

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "paper_id": "web",
                        "section": "web"
                    }
                )
            )

        return documents