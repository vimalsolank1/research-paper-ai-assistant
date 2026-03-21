<div align="center">

# 📚 Research Paper Assistant

### AI-powered platform to analyze, search, and interact with research papers

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)

<div align="center">

**Name** — Vimal  
**Email** — vimal162002@gmail.com  
**Live Demo** — [Click here](https://your-app.streamlit.app)  
**Video Explanation** — [Click here](https://your-video-link)

</div>

</div>

---

## What is this?

Research Paper Assistant lets you upload academic PDFs and interact with them using AI. Ask questions about a single paper, search across your entire library, compare multiple papers, and visualize trends — all powered by RAG, FAISS, and Groq LLM.

---

## How it works

```
PDFs → Section Detection → Chunking → Embeddings → FAISS Index
                                                         ↓
      User Query → Semantic Search → Relevant Chunks → LLM → Answer
```

Papers are split into meaningful sections (Abstract, Introduction, Method, Results, Conclusion) rather than fixed character chunks — giving much better retrieval accuracy for academic content.

---

## Features

- 💬 **Single Paper Chat** — Ask questions scoped to one selected paper
- 📚 **Library Chat** — Query across all papers with Doc / Web / Hybrid modes
- 🔀 **Compare Papers** — Auto-detects "compare/vs" queries across multiple papers
- 📊 **Trends Tab** — Keyword charts, publication trends, citation network
- 🔗 **Semantic Scholar** — Fetch real citation count, venue, DOI for any paper
- ⚡ **Fast Restarts** — FAISS index persisted to disk, no re-processing needed

---

## Getting Started

**1. Clone and install**
```bash
git clone https://github.com/your-username/research-paper-assistant.git
cd research-paper-assistant
pip install -r requirements.txt
```

**2. Create `.env` file**
```env
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
GPT_MODEL_NAME=llama-3.3-70b-versatile
TEMPERATURE=0.1
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
FAISS_INDEX_PATH=data/faiss_index
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
TOP_K_RESULTS=5
TOP_K_WEB_RESULTS=5
```

**3. Add your papers**
```
data/documents/
├── attention-is-all-you-need.pdf
├── bert-paper.pdf
└── any-paper.pdf
```

**4. Run**
```bash
streamlit run app.py
```

> First run builds the FAISS index (~1-2 min). Every restart after that loads from disk instantly.

---

## Project Structure

```
├── app.py                  # Entry point
├── config/settings.py      # Environment config
├── core/
│   ├── ingestion.py        # Section-based PDF chunking
│   ├── chain.py            # LangChain RAG pipeline
│   ├── vector_store.py     # FAISS store with persistence
│   ├── metadata_extractor.py  # Author, year, keyword extraction
│   └── schema.py           # Pydantic models (ResearchPaper, PaperSection, CitationRelationship)
├── tools/
│   ├── semantic_scholar.py # Semantic Scholar MCP tool
│   └── tavily_search.py    # Web search
├── ui/
│   ├── dashboard.py        # Dashboard tab
│   ├── chat.py             # Chat logic
│   ├── trends.py           # Trends tab
│   └── components.py       # Shared components
└── data/
    ├── documents/          # Put PDFs here
    └── faiss_index/        # Auto-generated
```

---

## Deployment on Streamlit Cloud

1. Push project to GitHub (include `data/documents/` with PDFs)
2. Go to [share.streamlit.io](https://share.streamlit.io) → connect repo → set `app.py` as entry point
3. Add all `.env` variables under **Settings → Secrets**

---

## Tech Stack

| | |
|---|---|
| LLM | Groq — LLaMA 3.3 70B |
| Embeddings | HuggingFace all-mpnet-base-v2 |
| Vector Store | FAISS |
| Framework | LangChain |
| Web Search | Tavily API |
| External API | Semantic Scholar |
| Frontend | Streamlit |

---

