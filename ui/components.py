"""
components.py — Shared UI components used across all tabs.

Includes session state setup, chat history, sidebar, and retrieval mode selector.
"""

import streamlit as st
from typing import List
import tempfile
import os


# ----------------------------------------
# SESSION STATE
# ----------------------------------------

def init_session_state():
    """Initialize all session state variables with default values on first load."""
    defaults = {
        "messages": [],
        "vector_store_initialized": False,
        "uploaded_files": [],
        "temp_dir": tempfile.mkdtemp(),
        "last_answer_meta": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ----------------------------------------
# CHAT HISTORY
# ----------------------------------------

def display_chat_history():
    """Render all previous messages in the Library Chat tab."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source}")


def add_message(role: str, content: str, sources: List[str] = None):
    """
    Append a message to the chat history.

    Args:
        role: "user" or "assistant".
        content: Message text.
        sources: Optional list of source strings to show under the message.
    """
    msg = {"role": role, "content": content}
    if sources:
        msg["sources"] = sources
    st.session_state.messages.append(msg)


def clear_chat_history():
    """Clear all messages from the Library Chat."""
    st.session_state.messages = []


# ----------------------------------------
# FILE SAVING
# ----------------------------------------

def save_uploaded_file(uploaded_file) -> str:
    """
    Save an uploaded file to the temp directory.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Path to the saved file.
    """
    file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# ----------------------------------------
# RETRIEVAL MODE SELECTOR
# ----------------------------------------

def retrieval_mode_selector() -> str:
    """
    Show a radio button for selecting retrieval mode.

    Returns:
        Selected mode string: "doc", "web", or "hybrid".
    """
    return st.radio(
        "Retrieval Mode",
        options=["doc", "web", "hybrid"],
        format_func=lambda x: {
            "doc": "📄 Papers",
            "web": "🌐 Web",
            "hybrid": "🔀 Hybrid"
        }[x],
        horizontal=True,
        index=0
    )


# ----------------------------------------
# SIDEBAR
# ----------------------------------------

def display_sidebar_info():
    """Render the sidebar with app info, usage guide, and paper list."""

    with st.sidebar:

        # App title with robot icon
        st.markdown("## 🤖 Research Paper Assistant")
        st.caption("AI-powered research intelligence platform")

        st.divider()

        # ── HOW TO USE ──────────────────────────────
        st.markdown("### 📖 How to Use")

        st.markdown("**📂 Papers Library**")
        st.markdown(
            "Browse all papers, view abstract, summary, "
            "citations and enrich metadata via Semantic Scholar."
        )

        st.markdown("**💬 Chat with a Paper**")
        st.markdown(
            "Select any paper → switch to *Chat* mode → "
            "ask questions scoped only to that paper."
        )

        st.markdown("**📚 Library Chat**")
        st.markdown(
            "Ask questions across **all papers** at once. "
            "Use **Compare** to compare two papers side by side."
        )

        st.markdown("**🔍 Retrieval Modes**")
        st.markdown(
            "- 📄 **Papers** — search only your uploaded papers  \n"
            "- 🌐 **Web** — search the internet via Tavily  \n"
            "- 🔀 **Hybrid** — combines both intelligently"
        )

        st.markdown("**📊 Trends**")
        st.markdown(
            "View keyword frequency, publication timeline, "
            "and citation network across your library."
        )

        st.divider()

        # ── LIBRARY PAPERS ───────────────────────────
        st.markdown("### 📚 Library Papers")

        if st.session_state.get("uploaded_files"):
            for file in st.session_state.uploaded_files:
                st.write(f"📄 {file}")
        else:
            st.caption("No papers loaded yet.")

        st.divider()

        # ── ACTIONS ──────────────────────────────────
        if st.button("🗑 Clear Library Chat", use_container_width=True):
            clear_chat_history()
            st.rerun()


# ----------------------------------------
# MISC — kept for compatibility
# ----------------------------------------

def display_answer_metadata():
    """Placeholder — not used currently."""
    pass


def display_processing_status(message: str, status: str = "info"):
    """
    Show a status message.

    Args:
        message: Text to display.
        status: One of "success", "warning", "error", "info".
    """
    if status == "success":
        st.success(message)
    elif status == "warning":
        st.warning(message)
    elif status == "error":
        st.error(message)
    else:
        st.info(message)