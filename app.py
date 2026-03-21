"""
app.py — Main Streamlit entry point.

Handles app config, library loading, and tab routing.
Three tabs: Papers Library, Library Chat, Trends.
"""

import json
import streamlit as st
from pathlib import Path

from ui.chat import ChatInterface
from ui.dashboard import render_dashboard
from ui.trends import render_trends
from ui.components import (
    init_session_state,
    display_chat_history,
    display_sidebar_info,
    add_message,
    retrieval_mode_selector,
    display_answer_metadata,
)

from config.settings import settings

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="🤖",
    layout="wide"
)

LIBRARY_DIR = Path("data/documents")
FAISS_INDEX_PATH = settings.FAISS_INDEX_PATH

# chunk counts saved here alongside FAISS index
CHUNK_COUNT_FILE = Path(FAISS_INDEX_PATH) / "chunk_counts.json"


# ── LIBRARY LOADER ────────────────────────────────────────────────────────────

def auto_load_library(chat_interface: ChatInterface):
    """
    Load papers and FAISS index on app startup.

    Fast path: if FAISS index already exists on disk, load it directly.
    Full path: first-time run — process all PDFs, embed, and save index.

    Args:
        chat_interface: The main ChatInterface instance from session state.
    """

    if st.session_state.get("library_loaded"):
        return

    faiss_path = Path(FAISS_INDEX_PATH)
    faiss_exists = (
        (faiss_path / "index.faiss").exists() and
        (faiss_path / "index.pkl").exists()
    )

    if not LIBRARY_DIR.exists():
        LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(LIBRARY_DIR.glob("*.pdf")) + list(LIBRARY_DIR.glob("*.txt"))

    from core.metadata_extractor import MetadataExtractor
    extractor = MetadataExtractor()

    # Initialize LLM for author extraction during metadata processing
    llm = None
    if pdf_files:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=settings.GPT_MODEL_NAME,
                temperature=settings.TEMPERATURE,
                api_key=settings.GROQ_API_KEY
            )
        except Exception:
            llm = None

    # ── FAST PATH — load saved FAISS index ───────────────────────────────────
    if faiss_exists and pdf_files:
        try:
            with st.spinner("Loading paper library..."):

                chat_interface.vector_store.load(FAISS_INDEX_PATH)
                st.session_state.vector_store_initialized = True

                # Load saved chunk counts from JSON file saved alongside FAISS
                # Without this, fast path always shows 0 chunks
                chunk_counts = {}
                if CHUNK_COUNT_FILE.exists():
                    with open(CHUNK_COUNT_FILE, "r") as f:
                        chunk_counts = json.load(f)

                for pdf_path in pdf_files:
                    paper_id = pdf_path.name

                    if paper_id not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(paper_id)

                    if paper_id not in st.session_state.paper_registry:
                        paper = extractor.extract(
                            file_path=str(pdf_path),
                            paper_id=paper_id,
                            # Read saved chunk count — not 0 anymore
                            chunk_count=chunk_counts.get(paper_id, 0),
                            llm=llm
                        )
                        st.session_state.paper_registry[paper_id] = paper

            st.session_state.library_loaded = True
            return

        except Exception as e:
            st.warning(f"Could not load saved index ({e}). Re-processing...")

    # ── FULL PATH — first time processing ────────────────────────────────────
    if not pdf_files:
        st.session_state.library_loaded = True
        return

    with st.spinner(f"Processing {len(pdf_files)} papers for the first time..."):

        all_chunks = []

        for pdf_path in pdf_files:
            paper_id = pdf_path.name

            if paper_id in st.session_state.get("uploaded_files", []):
                continue

            try:
                documents = chat_interface.doc_processor.process(str(pdf_path))
                all_chunks.extend(documents)
                st.session_state.uploaded_files.append(paper_id)

                if paper_id not in st.session_state.paper_registry:
                    paper = extractor.extract(
                        file_path=str(pdf_path),
                        paper_id=paper_id,
                        chunk_count=len(documents),
                        llm=llm
                    )
                    st.session_state.paper_registry[paper_id] = paper

            except Exception as e:
                st.warning(f"Could not load {paper_id}: {e}")

        if all_chunks:
            chat_interface.vector_store.add_documents(all_chunks)
            st.session_state.vector_store_initialized = True

            try:
                chat_interface.vector_store.save(FAISS_INDEX_PATH)

                # Save chunk count per paper alongside FAISS index
                # So fast path can read correct counts on next restart
                chunk_counts = {
                    paper_id: paper.chunk_count
                    for paper_id, paper in st.session_state.paper_registry.items()
                }
                with open(CHUNK_COUNT_FILE, "w") as f:
                    json.dump(chunk_counts, f)

            except Exception as e:
                st.warning(f"Could not save index: {e}")

    st.session_state.library_loaded = True


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():

    init_session_state()

    for key, default in {
        "paper_registry": {},
        "selected_paper_id": None,
        "chat_mode": "library",
        "library_loaded": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if "chat_interface" not in st.session_state:
        st.session_state.chat_interface = ChatInterface()

    chat_interface: ChatInterface = st.session_state.chat_interface

    auto_load_library(chat_interface)

    if st.session_state.vector_store_initialized:
        chat_interface.initialize_rag_chain()

    # Sidebar with usage guide and paper list
    display_sidebar_info()

    # Main header
    st.title("🤖 Research Paper Assistant")
    st.caption("Analyze, search, and interact with your research paper library using AI.")
    st.divider()

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📂  Papers Library",
        "📚  Library Chat",
        "📊  Trends & Insights"
    ])

    # ── TAB 1 — PAPERS LIBRARY ───────────────────────────────────────────────
    with tab1:
        render_dashboard(
            papers=st.session_state.paper_registry,
            chat_interface=chat_interface
        )

    # ── TAB 2 — LIBRARY CHAT ─────────────────────────────────────────────────
    with tab2:

        st.subheader("💬 Library Chat")
        st.caption(
            "Ask questions across all papers. "
            "Use **compare** or **vs** in your question to compare two papers."
        )

        retrieval_mode = retrieval_mode_selector()
        st.divider()

        display_chat_history()

        user_query = st.chat_input("Ask anything about your paper library...")

        if user_query:
            st.session_state.chat_mode = "library"
            add_message("user", user_query)
            full_response = ""

            with st.chat_message("assistant"):
                placeholder = st.empty()
                for token in chat_interface.get_response(user_query, retrieval_mode):
                    full_response += token
                    placeholder.markdown(full_response)

            add_message(
                "assistant",
                full_response,
                chat_interface.get_sources(user_query, retrieval_mode)
            )

            display_answer_metadata()
            st.rerun()

    # ── TAB 3 — TRENDS ───────────────────────────────────────────────────────
    with tab3:
        render_trends(st.session_state.paper_registry)


if __name__ == "__main__":
    main()