import streamlit as st
from typing import Dict
from collections import Counter
from core.schema import ResearchPaper
import pandas as pd


def render_trends(papers: Dict[str, ResearchPaper]):
    """
    Tab 3 — Trends & Insights .

    Shows:
    1. Top keywords bar chart
    2. Papers by year bar chart
    3. Most referenced papers table
    4. Citation network
    """

    if not papers:
        st.info("No papers in library. Add PDFs to data/documents/ and restart.")
        return

    paper_list = list(papers.values())

    st.subheader("Library Overview")

    # ----------------------------------------
    # 1. TOP KEYWORDS
    # ----------------------------------------

    st.markdown("#### Top Keywords")
    st.caption("Most common keywords across all papers in the library")

    all_keywords = []
    for p in paper_list:
        if p.keywords:
            all_keywords.extend(p.keywords)
        if p.abstract:
            stopwords = {
                "a", "an", "the", "of", "in", "for", "on", "with",
                "and", "is", "are", "was", "we", "our", "this", "that",
                "to", "from", "by", "as", "at", "be", "it", "its",
                "also", "which", "these", "their", "have", "has", "been",
                "can", "show", "used", "using", "based", "such", "paper",
                "model", "models", "learning", "neural", "deep", "approach"
            }
            words = p.abstract.lower().split()
            meaningful = [
                w.strip(".,;:()[]") for w in words
                if len(w) > 4 and w.lower() not in stopwords
            ]
            all_keywords.extend(meaningful[:10])

    if all_keywords:
        keyword_counts = Counter(all_keywords).most_common(15)
        kw_df = pd.DataFrame(keyword_counts, columns=["keyword", "count"]).set_index("keyword")
        st.bar_chart(kw_df["count"])
    else:
        st.caption("No keywords extracted yet.")

    st.divider()

    # ----------------------------------------
    # 2. PAPERS BY YEAR
    # ----------------------------------------

    st.markdown("#### Papers by Year")
    st.caption("Number of papers published per year in your library")

    years = [p.year for p in paper_list if p.year]

    if years:
        year_counts = Counter(years)
        sorted_years = sorted(year_counts.items())
        year_df = pd.DataFrame(sorted_years, columns=["year", "papers"])
        year_df["year"] = year_df["year"].astype(str)
        year_df = year_df.set_index("year")
        st.bar_chart(year_df["papers"])
    else:
        st.caption("No year data available.")

    st.divider()

    # ----------------------------------------
    # 3. MOST REFERENCED PAPERS
    # ----------------------------------------

    st.markdown("#### Most Referenced Papers")
    st.caption("Papers sorted by number of references extracted")

    ref_data = []
    for p in paper_list:
        ref_data.append({
            "Paper": p.title[:60] + "..." if len(p.title) > 60 else p.title,
            "Year": str(p.year) if p.year else "N/A",
            "Authors": ", ".join(p.authors[:2]) if p.authors else "N/A",
            "References": len(p.references)
        })

    ref_data_sorted = sorted(ref_data, key=lambda x: x["References"], reverse=True)
    st.dataframe(pd.DataFrame(ref_data_sorted), use_container_width=True, hide_index=True)

    st.divider()

    # ----------------------------------------
    # 4. CITATION NETWORK
    # ----------------------------------------

    st.markdown("#### Citation Network")
    st.caption("References extracted from each paper's reference section")

    for paper in paper_list:
        ref_count = len(paper.references)
        label = f"{paper.title[:60]} ({ref_count} references)"

        with st.expander(label):
            if not paper.references:
                st.caption("No references extracted from this paper.")
                continue

            library_titles = {
                p2.title.lower()[:40]: p2.title
                for p2 in paper_list
                if p2.paper_id != paper.paper_id and p2.title != "Unknown Title"
            }

            internal_refs = []
            external_refs = []

            for ref in paper.references:
                matched = False
                for lib_title_lower, lib_title in library_titles.items():
                    if lib_title_lower[:25] in ref.lower():
                        internal_refs.append((ref, lib_title))
                        matched = True
                        break
                if not matched:
                    external_refs.append(ref)

            if internal_refs:
                st.markdown("**Cites papers in your library:**")
                for ref, matched_title in internal_refs:
                    st.markdown(f"- ✅ **{matched_title}**")

            if external_refs:
                st.markdown(f"**External references ({len(external_refs)}):**")
                for ref in external_refs[:15]:
                    st.markdown(f"- {ref[:150]}")