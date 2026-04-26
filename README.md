# 🧠 DocuMind — Production RAG System

Chat with your documents using state-of-the-art Retrieval-Augmented Generation.

## 🚀 Features
- **Multi-format**: PDF, DOCX, TXT ingestion
- **Smart chunking**: Metadata-aware (source + page number)
- **Local embeddings**: HuggingFace `all-MiniLM-L6-v2` — no API cost
- **MMR retrieval**: Maximal Marginal Relevance for diverse results
- **Cross-encoder re-ranking**: `ms-marco-MiniLM` for precision
- **Groq LLM**: LLaMA 3.1 / LLaMA 3.3 / Gemma2 with streaming
- **Conversational memory**: Multi-turn context awareness
- **Source citations**: Every answer cites its source page
- **Chat export**: Download full conversation as .txt
- **Model selector**: Switch LLMs from the sidebar

## 🛠️ Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/yourusername/documind-rag
cd documind-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY from https://console.groq.com

# 5. Run
streamlit run app.py
```

## 🏗️ Architecture

```
Documents (PDF/DOCX/TXT)
        ↓
   Text Extraction + Page Metadata
        ↓
   RecursiveCharacterTextSplitter (800 chars, 150 overlap)
        ↓
   HuggingFace all-MiniLM-L6-v2 Embeddings (local CPU)
        ↓
   FAISS Vector Store (persisted to disk)
        ↓
   User Query
        ↓
   MMR Retrieval (top-10 diverse chunks)
        ↓
   CrossEncoder Re-ranking (ms-marco → top-4)
        ↓
   Groq LLaMA-3.1 with chat history context
        ↓
   Streamed answer + source citations
```

## 🧰 Tech Stack
| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq (LLaMA 3.1 / 3.3 / Gemma2) |
| Embeddings | HuggingFace sentence-transformers |
| Vector DB | FAISS |
| Re-ranker | CrossEncoder ms-marco-MiniLM |
| Orchestration | LangChain LCEL |

## 📄 License
MIT
# DocuMind
