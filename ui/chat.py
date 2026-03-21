from typing import Generator, Optional
import streamlit as st

from core.ingestion import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.chain import RAGChain
from tools.tavily_search import TavilySearchTool
from ui.components import save_uploaded_file


class ChatInterface:

    def __init__(self):
        # initialize all core components
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        self.rag_chain: Optional[RAGChain] = None
        self.tavily = TavilySearchTool()

    # ----------------------------------------
    # DOCUMENT PROCESSING
    # ----------------------------------------

    def process_uploaded_files(self, uploaded_files, paper_registry: dict = None) -> int:

        # extract metadata
        from core.metadata_extractor import MetadataExtractor
        extractor = MetadataExtractor()
        all_chunks = []

        for uploaded_file in uploaded_files:
            # save uploaded file
            file_path = save_uploaded_file(uploaded_file)

            # convert pdf into chunks
            documents = self.doc_processor.process(file_path)
            all_chunks.extend(documents)

            # track uploaded files
            if uploaded_file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file.name)

            # store metadata in registry
            if paper_registry is not None and uploaded_file.name not in paper_registry:
                paper = extractor.extract(
                    file_path=file_path,
                    paper_id=uploaded_file.name,
                    chunk_count=len(documents)
                )
                paper_registry[uploaded_file.name] = paper

        # add chunks to vector store
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            st.session_state.vector_store_initialized = True

        return len(all_chunks)

    # ----------------------------------------
    # INIT RAG
    # ----------------------------------------

    def initialize_rag_chain(self):
        # initialize rag only once
        if self.vector_store.is_initialized and self.rag_chain is None:
            self.rag_chain = RAGChain(self.vector_store)

    # ----------------------------------------
    # RETRIEVAL
    # ----------------------------------------

    def retrieve_documents(self, query: str, force_all: bool = False):

        # return empty if vector store not ready
        if not self.vector_store.is_initialized:
            return []

        # search top documents
        results = self.vector_store.search_with_scores(query, k=20)
        results = sorted(results, key=lambda x: x[1])

        chat_mode = st.session_state.get("chat_mode", "library")
        selected_id = st.session_state.get("selected_paper_id")

        # filter for single paper mode
        if not force_all and chat_mode == "single" and selected_id:
            results = [
                (doc, score) for doc, score in results
                if doc.metadata.get("source") == selected_id
            ]

        return results[:5]

    # ----------------------------------------
    # HELPERS
    # ----------------------------------------

    def _get_selected_paper(self):
        # get selected paper from session
        selected_id = st.session_state.get("selected_paper_id")
        if not selected_id:
            return None
        return st.session_state.get("paper_registry", {}).get(selected_id)

    def _set_paper_source(self, section: str = ""):
        # store source info for answer tracking
        selected_id = st.session_state.get("selected_paper_id", "")
        if selected_id:
            from langchain_core.documents import Document
            dummy = Document(
                page_content="",
                metadata={"source": selected_id, "section": section}
            )
            st.session_state.last_answer_meta["doc_chunks"] = [dummy]
            st.session_state.last_answer_meta["answer_type"] = "doc"

    def _is_metadata_question(self, query: str) -> Optional[str]:

        # handle simple metadata queries without LLM
        chat_mode = st.session_state.get("chat_mode", "library")
        if chat_mode != "single":
            return None

        paper = self._get_selected_paper()
        if not paper:
            return None

        q = query.lower()

        # author info
        if any(w in q for w in ["author", "who wrote", "who published", "written by"]):
            if paper.authors:
                return f"The authors of this paper are: **{', '.join(paper.authors)}**"
            return "Author information was not extracted from this paper."

        # publication year
        if any(w in q for w in ["year", "when was", "published in", "publication year"]):
            if paper.year:
                return f"This paper was published in **{paper.year}**."
            return "Publication year was not extracted from this paper."

        # venue info
        if any(w in q for w in ["venue", "conference", "journal", "where was published"]):
            if paper.venue:
                return f"This paper was published at **{paper.venue}**."
            return "Venue information was not extracted from this paper."

        # title
        if any(w in q for w in ["title", "name of this paper", "what is this paper called"]):
            return f"The title is: **{paper.title}**"

        # keywords
        if any(w in q for w in ["keyword", "topics of"]):
            if paper.keywords:
                return f"Keywords: **{', '.join(paper.keywords)}**"

        return None

    def _get_section_content(self, section: str) -> Optional[str]:

        # get section text from selected paper
        chat_mode = st.session_state.get("chat_mode", "library")
        if chat_mode != "single":
            return None

        paper = self._get_selected_paper()
        if not paper:
            return None

        sec = paper.get_section(section)
        if sec and sec.content:
            return sec.content

        # try matching similar section names
        aliases = {
            "method": ["method", "methodology", "model", "approach", "architecture"],
            "results": ["results", "experiments", "evaluation"],
            "related_work": ["related_work", "background"],
            "abstract": ["abstract"],
            "introduction": ["introduction"],
            "conclusion": ["conclusion", "concluding"],
        }

        for alias in aliases.get(section, [section]):
            for s in paper.sections:
                if alias in s.name.lower():
                    return s.content

        return None

    def _get_chunks_by_section(self, section: str):

        # retrieve chunks based on section
        if not self.vector_store.is_initialized:
            return []

        results = self.vector_store.search_with_scores(section, k=30)

        chat_mode = st.session_state.get("chat_mode", "library")
        selected_id = st.session_state.get("selected_paper_id")

        # filter for selected paper
        if chat_mode == "single" and selected_id:
            results = [
                (doc, score) for doc, score in results
                if doc.metadata.get("source") == selected_id
            ]

        # match exact section
        tagged = [
            doc for doc, _ in results
            if doc.metadata.get("section", "").lower() == section.lower()
        ]

        if tagged:
            return tagged[:8]

        # fallback using page order
        sorted_by_page = sorted(results, key=lambda x: x[0].metadata.get("page", 999))

        if section == "abstract":
            return [doc for doc, _ in sorted_by_page[:3]]
        elif section == "conclusion":
            return [doc for doc, _ in sorted_by_page[-3:]]
        elif section == "introduction":
            return [doc for doc, _ in sorted_by_page[1:5]]
        else:
            return [doc for doc, _ in sorted_by_page[:5]]

    def _get_fallback_context(self) -> Optional[str]:

        # fallback if retrieval fails
        chat_mode = st.session_state.get("chat_mode", "library")
        if chat_mode != "single":
            return None

        paper = self._get_selected_paper()
        if not paper:
            return None

        parts = []
        abstract = self._get_section_content("abstract")
        if abstract:
            parts.append(f"Abstract:\n{abstract}")

        intro = self._get_section_content("introduction")
        if intro:
            parts.append(f"Introduction:\n{intro[:2000]}")

        return "\n\n".join(parts) if parts else None

    def _is_paper_related(self, query: str) -> bool:
        # check if query is about paper content
        keywords = [
            "paper", "abstract", "introduction", "method", "result",
            "conclusion", "author", "figure", "table", "section",
            "study", "research", "experiment", "model", "dataset",
            "approach", "propose", "algorithm", "architecture",
            "training", "accuracy", "transformer", "attention"
        ]
        return any(k in query.lower() for k in keywords)

    def _is_compare_query(self, query: str) -> bool:
        # detect comparison queries
        compare_words = [
            "compare", "comparison", "difference", "vs", "versus",
            "contrast", "similar", "differ", "both papers", "all papers"
        ]
        return any(k in query.lower() for k in compare_words)

    # ----------------------------------------
    # MAIN RESPONSE
    # ----------------------------------------

    def get_response(self, query: str, retrieval_mode: str) -> Generator[str, None, None]:

        # initialize rag if needed
        if self.rag_chain is None and self.vector_store.is_initialized:
            self.initialize_rag_chain()

        query_lower = query.lower()
        chat_mode = st.session_state.get("chat_mode", "library")
        selected_id = st.session_state.get("selected_paper_id")

        # store sources info
        st.session_state.last_answer_meta = {
            "answer_type": "doc",
            "doc_chunks": [],
            "web_docs": [],
        }

        # metadata questions shortcut
        metadata_answer = self._is_metadata_question(query)
        if metadata_answer:
            self._set_paper_source("metadata")
            yield metadata_answer
            return

        # compare multiple papers
        if self._is_compare_query(query) and self.vector_store.is_initialized:

            all_results = self.vector_store.search_with_scores(query, k=30)
            all_results = sorted(all_results, key=lambda x: x[1])

            paper_chunks = {}
            for doc, score in all_results[:20]:
                src = doc.metadata.get("source", "unknown")
                if src not in paper_chunks:
                    paper_chunks[src] = []
                paper_chunks[src].append(doc)

            if len(paper_chunks) < 2:
                yield "Please add at least 2 papers to compare."
                return

            context_parts = []
            docs_used = []
            paper_names = {}

            for i, (paper_name, chunks) in enumerate(list(paper_chunks.items())[:4]):
                short_name = f"Paper {i+1} ({paper_name[:30]})"
                paper_names[paper_name] = short_name

                context_parts.append(
                    f"[{short_name}]\n" +
                    "\n".join([c.page_content for c in chunks[:3]])
                )
                docs_used.extend(chunks[:2])

            st.session_state.last_answer_meta["doc_chunks"] = docs_used
            context = "\n\n".join(context_parts)

            prompt = f"""
You are a research assistant. Compare the following research papers based on the context below.

Important rules:
- Reference papers by their name naturally in your answer (e.g. "Paper 1", "Paper 2")
- Do NOT copy or show the internal labels like [Paper 1 (...)] in your answer
- Do NOT show raw formulas or equations directly
- Give a clean structured comparison with clear headings
- For each comparison point, clearly state which paper does what

Context:
{context}

Question: {query}

Write a clean structured comparison:
"""
            yield self.rag_chain.llm.invoke(prompt).content
            return

        # summary mode
        if any(x in query_lower for x in ["summary", "summarize", "summarise"]):

            abstract_content = self._get_section_content("abstract")

            if abstract_content:
                self._set_paper_source("abstract")
                context = abstract_content
            else:
                docs = self.vector_store.search(query, k=20)
                if chat_mode == "single" and selected_id:
                    docs = [d for d in docs if d.metadata.get("source") == selected_id]
                if not docs:
                    yield "No content found to summarize."
                    return
                st.session_state.last_answer_meta["doc_chunks"] = docs
                context = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
Summarize this research paper using ONLY the given text.

Text:
{context}

Give:
- Problem
- Method
- Result
- Conclusion
"""
            yield self.rag_chain.llm.invoke(prompt).content
            return

        # section based queries
        sections = ["abstract", "introduction", "method", "results", "conclusion"]

        for sec in sections:
            if sec in query_lower:

                section_text = self._get_section_content(sec)

                if section_text:
                    self._set_paper_source(sec)
                    prompt = f"""
You are a research assistant. Explain the {sec} section of this research paper.

Rules:
- Use ONLY the given text
- Do NOT add general theory

Text:
{section_text}

Explain the {sec}:
"""
                    yield self.rag_chain.llm.invoke(prompt).content
                    return

                docs = self._get_chunks_by_section(sec)

                if not docs:
                    yield f"The {sec} section was not found in this paper."
                    return

                st.session_state.last_answer_meta["doc_chunks"] = docs
                context = "\n\n".join([d.page_content for d in docs])

                prompt = f"""
You are a research assistant. Extract and explain the {sec} section.

Rules:
- Use ONLY the given text
- Do NOT add general theory

Text:
{context}

Explain the {sec}:
"""
                yield self.rag_chain.llm.invoke(prompt).content
                return

        # document QA
        if retrieval_mode == "doc":

            results = self.retrieve_documents(query)
            docs = [doc for doc, _ in results]

            if not docs and chat_mode == "single":
                fallback = self._get_fallback_context()
                if fallback:
                    self._set_paper_source("abstract")
                    prompt = f"""
You are a research assistant. Answer the question using ONLY the paper context below.

Paper context:
{fallback}

Question: {query}

Answer specifically based on the paper:
"""
                    yield self.rag_chain.llm.invoke(prompt).content
                    return
                else:
                    yield "No relevant content found in this paper."
                    return

            if not docs:
                yield "No relevant content found in the paper library."
                return

            st.session_state.last_answer_meta["doc_chunks"] = docs

            if chat_mode == "single" and selected_id:
                context = "\n\n".join([
                    f"[Section: {d.metadata.get('section', 'unknown')}]\n{d.page_content}"
                    for d in docs
                ])
                prompt = f"""
You are a research assistant. Answer using ONLY the paper content below.

Context:
{context}

Question: {query}

Answer:
"""
                yield self.rag_chain.llm.invoke(prompt).content
            else:
                for token in self.rag_chain.query_stream(query):
                    yield token

            return

        # web QA
        if retrieval_mode == "web":

            web_docs = self.tavily.search(query)

            if not web_docs:
                yield "No useful web results found."
                return

            st.session_state.last_answer_meta["web_docs"] = web_docs
            st.session_state.last_answer_meta["answer_type"] = "web"
            context = "\n\n".join([w.page_content for w in web_docs[:5]])

            prompt = f"""
Answer the question using the web results below.

Context:
{context}

Question: {query}
"""
            yield self.rag_chain.llm.invoke(prompt).content
            return

        # hybrid mode
        if retrieval_mode == "hybrid":

            paper_related = self._is_paper_related(query)
            realtime = any(w in query_lower for w in ["news", "latest", "today", "current", "recent"])

            context_parts = []

            if paper_related and self.vector_store.is_initialized:
                results = self.retrieve_documents(query)
                docs = [doc for doc, _ in results]
                st.session_state.last_answer_meta["doc_chunks"] = docs
                for doc in docs[:3]:
                    context_parts.append(doc.page_content)

            if realtime or not paper_related:
                web_docs = self.tavily.search(query)
                st.session_state.last_answer_meta["web_docs"] = web_docs
                for w in web_docs[:3]:
                    context_parts.append(w.page_content)

            if not context_parts:
                yield "No relevant content found."
                return

            used_paper = bool(st.session_state.last_answer_meta["doc_chunks"])
            used_web = bool(st.session_state.last_answer_meta["web_docs"])

            if used_paper and used_web:
                st.session_state.last_answer_meta["answer_type"] = "hybrid"
            elif used_paper:
                st.session_state.last_answer_meta["answer_type"] = "doc"
            else:
                st.session_state.last_answer_meta["answer_type"] = "web"

            context = "\n\n".join(context_parts)

            prompt = f"""
Answer using available context. Prefer paper content if relevant.

Context:
{context}

Question: {query}
"""
            yield self.rag_chain.llm.invoke(prompt).content

    # ----------------------------------------
    # SOURCES
    # ----------------------------------------

    def get_sources(self, query: str, retrieval_mode: str):

        # return sources used in answer
        meta = st.session_state.get("last_answer_meta", {})
        if not meta:
            return []

        sources = set()
        answer_type = meta.get("answer_type", retrieval_mode)

        # paper sources
        if answer_type in ("doc", "hybrid"):
            for doc in meta.get("doc_chunks", []):
                src = doc.metadata.get("source", "unknown")
                section = doc.metadata.get("section", "")
                if section and section not in ("unknown", ""):
                    sources.add(f"[Paper] {src} — {section}")
                else:
                    sources.add(f"[Paper] {src}")

        # web sources
        if answer_type in ("web", "hybrid"):
            for web in meta.get("web_docs", []):
                src = web.metadata.get("source", "")
                if src:
                    sources.add(f"[Web] {src}")

        return list(sources)