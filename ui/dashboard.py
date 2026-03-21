import streamlit as st
from typing import Dict
from core.schema import ResearchPaper


def render_dashboard(papers: Dict[str, ResearchPaper], chat_interface=None):
    """
    Tab 1 — Dashboard.
    LEFT  : Paper library cards (filter, select one paper)
    RIGHT : Paper viewer + single-paper chatbot

    MCP Tool  (Paper Metadata Lookup) is in the Paper Viewer section.
    """

    if not papers:
        st.info("No papers in library. Add PDFs to data/documents/ and restart the app.")
        return

    paper_list = list(papers.values())

    # ----------------------------------------
    # STATS ROW
    # ----------------------------------------

    c1, c2, c3, c4 = st.columns(4)
    years = [p.year for p in paper_list if p.year]
    c1.metric("Total Papers", len(paper_list))
    c2.metric("Year Range", f"{min(years)}–{max(years)}" if len(years) > 1 else (str(years[0]) if years else "N/A"))
    c3.metric("Total Chunks", sum(p.chunk_count for p in paper_list))
    c4.metric("Total References", sum(len(p.references) for p in paper_list))

    st.divider()

    # ----------------------------------------
    # FILTERS
    # ----------------------------------------

    st.subheader("Filters")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        all_years = sorted(set(p.year for p in paper_list if p.year))
        if len(all_years) >= 2:
            year_range = st.slider("Year range", min_value=min(all_years), max_value=max(all_years), value=(min(all_years), max(all_years)))
        elif len(all_years) == 1:
            year_range = (all_years[0], all_years[0])
            st.caption(f"Year: {all_years[0]}")
        else:
            year_range = None

    with fc2:
        all_keywords = sorted(set(kw for p in paper_list for kw in p.keywords))
        selected_keywords = st.multiselect("Topic / keyword", options=all_keywords)

    with fc3:
        all_venues = sorted(set(p.venue for p in paper_list if p.venue))
        selected_venues = st.multiselect("Venue", options=all_venues)

    filtered = paper_list
    if year_range:
        filtered = [p for p in filtered if p.year and year_range[0] <= p.year <= year_range[1]]
    if selected_keywords:
        filtered = [p for p in filtered if any(kw in p.keywords for kw in selected_keywords)]
    if selected_venues:
        filtered = [p for p in filtered if p.venue in selected_venues]

    st.caption(f"Showing {len(filtered)} of {len(paper_list)} papers")
    st.divider()

    # ----------------------------------------
    # TWO COLUMN LAYOUT
    # ----------------------------------------

    col_lib, col_right = st.columns([1, 1], gap="large")

    with col_lib:
        st.subheader("Paper Library")
        if not filtered:
            st.warning("No papers match your filters.")
        else:
            for paper in filtered:
                _render_paper_card(paper)

    with col_right:
        selected_id = st.session_state.get("selected_paper_id")

        if not selected_id or selected_id not in papers:
            st.subheader("Paper Viewer")
            st.info("Click any paper on the left to view details and chat with it.")
        else:
            selected_paper = papers[selected_id]

            view_mode = st.radio(
                "View mode",
                options=["📄 Paper Info", "💬 Chat with this paper"],
                horizontal=True,
                key="dashboard_view_mode"
            )

            st.divider()

            if view_mode == "📄 Paper Info":
                _render_paper_viewer(selected_paper, chat_interface)
            else:
                _render_paper_chat(selected_paper, chat_interface)


# ----------------------------------------
# PAPER CARD
# ----------------------------------------

def _render_paper_card(paper: ResearchPaper):

    is_selected = st.session_state.get("selected_paper_id") == paper.paper_id

    with st.container(border=True):
        title = paper.title[:65] + "..." if len(paper.title) > 65 else paper.title
        if is_selected:
            st.markdown(f"**✅ {title}**")
        else:
            st.markdown(f"**{title}**")

        meta = []
        if paper.authors:
            meta.append(", ".join(paper.authors[:2]))
        if paper.year:
            meta.append(str(paper.year))
        if paper.venue:
            meta.append(paper.venue)
        st.caption(" · ".join(meta) if meta else "No metadata")

        if paper.keywords:
            st.markdown("  ".join([f"`{kw}`" for kw in paper.keywords[:4]]))

        btn_label = "✅ Selected" if is_selected else "Select"
        if st.button(btn_label, key=f"select_{paper.paper_id}", use_container_width=True):
            st.session_state.selected_paper_id = paper.paper_id
            st.session_state[f"paper_messages_{paper.paper_id}"] = []
            st.rerun()


# ----------------------------------------
# PAPER VIEWER — with MCP Tool 
# ----------------------------------------

def _render_paper_viewer(paper: ResearchPaper, chat_interface=None):

    st.subheader(paper.title[:60] + "..." if len(paper.title) > 60 else paper.title)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Year", paper.year or "N/A")
    mc2.metric("Chunks", paper.chunk_count)
    mc3.metric("Refs", len(paper.references))

    if paper.authors:
        st.markdown(f"**Authors:** {', '.join(paper.authors)}")
    if paper.venue:
        st.markdown(f"**Venue:** {paper.venue}")
    if paper.keywords:
        st.markdown(f"**Keywords:** {', '.join(paper.keywords)}")

    # ----------------------------------------
    # MCP TOOL  — Paper Metadata Lookup
    # ----------------------------------------

    st.divider()
    st.markdown("#### Enrich Metadata")
    st.caption("Fetch real citation count, venue and DOI from Semantic Scholar")

    enrich_key = f"enriched_{paper.paper_id}"

    if enrich_key in st.session_state:
        enriched = st.session_state[enrich_key]
        if enriched:
            col1, col2, col3 = st.columns(3)
            col1.metric("Citations", enriched.citation_count)
            col2.metric("Venue", enriched.venue or "N/A")
            col3.metric("Year (verified)", enriched.year or "N/A")
            if enriched.doi:
                st.markdown(f"**DOI:** `{enriched.doi}`")
            if enriched.url:
                st.markdown(f"**URL:** [{enriched.url}]({enriched.url})")
        else:
            st.warning("Paper not found on Semantic Scholar.")

        if st.button("Refresh", key=f"refresh_{paper.paper_id}"):
            del st.session_state[enrich_key]
            st.rerun()

    else:
        if st.button("🔍 Fetch from Semantic Scholar", key=f"enrich_{paper.paper_id}", type="primary"):
            with st.spinner("Looking up paper on Semantic Scholar..."):
                try:
                    from tools.semantic_scholar import SemanticScholarTools
                    tool = SemanticScholarTools()
                    result = tool.lookup_paper_metadata(paper.title)
                    st.session_state[enrich_key] = result

                    # FIX: always update registry with verified data from Semantic Scholar
                    # Semantic Scholar data is more accurate than PDF extraction
                    if result:
                        registry = st.session_state.get("paper_registry", {})
                        if paper.paper_id in registry:

                            # Always update year — Semantic Scholar year is verified
                            if result.year:
                                registry[paper.paper_id].year = result.year

                            # Update venue if we have one
                            if result.venue:
                                registry[paper.paper_id].venue = result.venue

                            # Update authors if Semantic Scholar found more complete list
                            if result.authors and len(result.authors) > len(registry[paper.paper_id].authors):
                                registry[paper.paper_id].authors = result.authors

                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "rate" in error_msg:
                        st.warning("Semantic Scholar rate limit reached. Please wait 5 minutes and try again.")
                    elif "timeout" in error_msg or "connection" in error_msg:
                        st.warning("Could not connect to Semantic Scholar. Check your internet connection.")
                    else:
                        st.error(f"Error: {e}")
                    st.session_state[enrich_key] = None

            st.rerun()

    st.divider()

    # Abstract
    st.markdown("#### Abstract")
    if paper.abstract:
        st.write(paper.abstract)
    else:
        st.caption("Abstract not extracted.")

    st.divider()

    # AI Summary
    st.markdown("#### Auto-generated Summary")
    summary_key = f"ai_summary_{paper.paper_id}"

    if summary_key in st.session_state:
        st.write(st.session_state[summary_key])
        if st.button("Regenerate", key=f"regen_{paper.paper_id}"):
            del st.session_state[summary_key]
            st.rerun()
    else:
        rag = chat_interface.rag_chain if chat_interface else None
        if rag and paper.abstract:
            if st.button("Generate Summary", key=f"gen_{paper.paper_id}", type="primary"):
                with st.spinner("Generating..."):
                    prompt = f"""
Generate a structured summary of this research paper.

Abstract:
{paper.abstract}

Format exactly as:
**Problem:** what problem this solves
**Approach:** what method is proposed
**Key Contributions:** what is new
**Results:** key results achieved
**Limitations:** any limitations noted

Use only the abstract above. Be concise and academic.
"""
                    try:
                        summary = rag.llm.invoke(prompt).content
                        st.session_state[summary_key] = summary
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
        else:
            st.caption("Process papers first to enable summary generation.")

    st.divider()

    # Citations
    st.markdown("#### Citation Information")
    if paper.references:
        st.caption(f"{len(paper.references)} references extracted")
        with st.expander("View references"):
            for i, ref in enumerate(paper.references[:25], 1):
                st.markdown(f"{i}. {ref}")
    else:
        st.caption("No references extracted.")


# ----------------------------------------
# SINGLE-PAPER CHATBOT
# ----------------------------------------

def _render_paper_chat(paper: ResearchPaper, chat_interface=None):

    st.subheader(f"💬 Chat: {paper.title[:45]}...")
    st.caption("Asking questions only about this paper")

    if not chat_interface:
        st.error("Chat interface not available.")
        return

    if not chat_interface.vector_store.is_initialized:
        st.warning("Papers not indexed yet.")
        return

    msg_key = f"paper_messages_{paper.paper_id}"
    if msg_key not in st.session_state:
        st.session_state[msg_key] = []

    for message in st.session_state[msg_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for src in message["sources"]:
                        st.write(f"- {src}")

    user_query = st.chat_input(
        f"Ask about {paper.title[:30]}...",
        key=f"chat_input_{paper.paper_id}"
    )

    if user_query:

        st.session_state[msg_key].append({"role": "user", "content": user_query})

        original_mode = st.session_state.get("chat_mode")
        original_paper = st.session_state.get("selected_paper_id")
        st.session_state.chat_mode = "single"
        st.session_state.selected_paper_id = paper.paper_id

        full_response = ""

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                for token in chat_interface.get_response(user_query, "doc"):
                    full_response += token
                    placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Error: {e}"
                placeholder.markdown(full_response)

        sources = chat_interface.get_sources(user_query, "doc")

        st.session_state[msg_key].append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })

        st.session_state.chat_mode = original_mode
        st.session_state.selected_paper_id = original_paper

        st.rerun()

    if st.session_state[msg_key]:
        if st.button("Clear chat", key=f"clear_{paper.paper_id}"):
            st.session_state[msg_key] = []
            st.rerun()