"""
chain.py — LangChain RAG pipeline for research paper Q&A.

Combines FAISS retrieval with Groq LLM to answer questions
grounded in the actual paper content.

Flow: query → retrieve chunks → build context → LLM → answer
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from config.settings import settings


# System prompt for the RAG chain.
# Instructs the LLM to answer only from provided context,
# and to reference the source paper and section in its answer.
RAG_PROMPT = """
You are a research assistant.

Answer the question using ONLY the provided context.
If the answer is not present, say:
"I could not find the answer in the provided documents."

Give clear and concise answers. When possible, refer to the paper and section.

Context:
{context}

Question:
{question}

Answer:
"""


class RAGChain:
    """
    Orchestrates the full RAG pipeline: retrieve → build context → generate.

    Uses:
    - VectorStoreManager for semantic chunk retrieval
    - Groq LLaMA 3.3 70B for answer generation
    - LangChain's pipe operator (|) to chain prompt → LLM → parser
    """

    def __init__(self, vector_store):
        """
        Args:
            vector_store: VectorStoreManager instance with an initialized index.
        """
        self.vector_store = vector_store

        self.llm = ChatGroq(
            model=settings.GPT_MODEL_NAME,
            temperature=settings.TEMPERATURE,
            api_key=settings.GROQ_API_KEY
        )

        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        self.parser = StrOutputParser()

        # LangChain pipe: prompt feeds into LLM, LLM output feeds into parser
        self.chain = self.prompt | self.llm | self.parser

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        Fetch the most relevant chunks for a query from FAISS.

        Args:
            query: User's question.
            k: Number of chunks to retrieve.

        Returns:
            List of matching Document chunks.
        """
        return self.vector_store.search(query, k=k)

    def _build_context(self, documents: List[Document]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM.

        Each chunk is labeled with its source paper and section so the LLM
        can reference them in its answer.

        Args:
            documents: Retrieved chunks from FAISS.

        Returns:
            Formatted multi-document context string.
        """
        if not documents:
            return "No relevant documents found."

        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            section = doc.metadata.get("section", "unknown")
            context_parts.append(
                f"[Doc {i}] Paper: {source} | Section: {section}\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)

    def generate(self, query: str, context: str) -> str:
        """
        Generate an answer from the LLM given a query and context.

        Args:
            query: User's question.
            context: Formatted context string from _build_context().

        Returns:
            LLM-generated answer string.
        """
        return self.chain.invoke({
            "context": context,
            "question": query
        })

    def query(self, query: str, k: int = None) -> str:
        """
        Run the full RAG pipeline and return a complete answer.

        Args:
            query: User's question.
            k: Number of chunks to retrieve.

        Returns:
            Final answer string.
        """
        docs = self.retrieve(query, k=k)
        context = self._build_context(docs)
        return self.generate(query, context)

    def query_stream(self, query: str, k: int = None):
        """
        Same as query() but streams tokens as they are generated.

        Used by the Streamlit UI to show responses word-by-word.

        Args:
            query: User's question.
            k: Number of chunks to retrieve.

        Yields:
            str: Individual tokens from the LLM response.
        """
        docs = self.retrieve(query, k=k)
        context = self._build_context(docs)

        for token in self.chain.stream({
            "context": context,
            "question": query
        }):
            yield token

    def summarize_documents(self, documents: List[Document], top_n: int = 3):
        """
        Generate a structured summary for each of the top N documents.

        Args:
            documents: Input documents to summarize.
            top_n: How many documents to summarize. Defaults to 3.

        Returns:
            List of summary strings, one per document.
        """
        summaries = []

        for i, doc in enumerate(documents[:top_n], 1):
            prompt = f"""
Summarize this research paper content:

Text:
{doc.page_content}

Return:
- Problem
- Approach
- Key result
"""
            summary = self.llm.invoke(prompt).content
            source = doc.metadata.get("source", "unknown")
            summaries.append(f"[Doc {i}] Paper: {source}\n{summary}")

        return summaries