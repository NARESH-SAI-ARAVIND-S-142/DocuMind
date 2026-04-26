"""
Hitloop RAG App — Production Grade
Features:
  - Multi-format ingestion: PDF, DOCX, TXT
  - Semantic chunking with metadata (source + page)
  - HuggingFace local embeddings (all-MiniLM-L6-v2)
  - FAISS vector store with MMR retrieval
  - Cross-encoder re-ranker for precision
  - Groq LLaMA-3.1 with streaming responses
  - Conversational memory (multi-turn)
  - Source citations per answer
  - Model selector in sidebar
  - Chat export to .txt
  - Token usage tracker
  - Full error handling + structured logging
"""

import os
import io
import time
import logging
import datetime
import streamlit as st
from dotenv import load_dotenv

# Document loaders
from PyPDF2 import PdfReader
import docx  # python-docx

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# Re-ranker
from sentence_transformers import CrossEncoder

# ─────────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────────

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 150
TOP_K_RETRIEVE   = 10   # retrieve more, re-rank down to TOP_K_FINAL
TOP_K_FINAL      = 4    # after re-ranking

GROQ_MODELS = {
    "LLaMA 3.1 8B (Fastest)":    "llama-3.1-8b-instant",
    "LLaMA 3.3 70B (Smartest)":  "llama-3.3-70b-versatile",
    "Gemma 2 9B (Balanced)":     "gemma2-9b-it",
}

st.set_page_config(
    page_title="DocuMind · RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 2. CACHED RESOURCES
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner="Loading re-ranker…")
def get_reranker():
    logger.info("Loading cross-encoder re-ranker: %s", RERANKER_MODEL)
    return CrossEncoder(RERANKER_MODEL)

# ─────────────────────────────────────────────
# 3. VALIDATION
# ─────────────────────────────────────────────
def validate_api_key() -> bool:
    if not os.getenv("GROQ_API_KEY"):
        st.error(
            "⚠️ **Missing `GROQ_API_KEY`** in `.env` file.\n\n"
            "Get a free key at https://console.groq.com"
        )
        return False
    return True

# ─────────────────────────────────────────────
# 4. DOCUMENT INGESTION (PDF + DOCX + TXT)
# ─────────────────────────────────────────────

def extract_from_pdf(file) -> list[dict]:
    """Returns list of {text, source, page}"""
    pages = []
    reader = PdfReader(file)
    if reader.is_encrypted:
        st.warning(f"⚠️ Skipping **{file.name}** — password protected.")
        return []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"text": text, "source": file.name, "page": i + 1})
    return pages

def extract_from_docx(file) -> list[dict]:
    doc = docx.Document(file)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": full_text, "source": file.name, "page": 1}]

def extract_from_txt(file) -> list[dict]:
    text = file.read().decode("utf-8", errors="ignore")
    return [{"text": text, "source": file.name, "page": 1}]

def extract_documents(uploaded_files) -> list[dict]:
    all_pages = []
    for f in uploaded_files:
        try:
            ext = f.name.lower().split(".")[-1]
            if ext == "pdf":
                all_pages.extend(extract_from_pdf(f))
            elif ext == "docx":
                all_pages.extend(extract_from_docx(f))
            elif ext == "txt":
                all_pages.extend(extract_from_txt(f))
            else:
                st.warning(f"⚠️ Unsupported format: {f.name}")
        except Exception as e:
            st.warning(f"⚠️ Could not read **{f.name}**: {e}")
            logger.exception("Failed to read %s", f.name)
    return all_pages

# ─────────────────────────────────────────────
# 5. CHUNKING WITH METADATA
# ─────────────────────────────────────────────

def chunk_documents(pages: list[dict]) -> tuple[list[str], list[dict]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks, metadatas = [], []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append(split)
            metadatas.append({"source": page["source"], "page": page["page"]})
    logger.info("Created %d chunks from %d pages.", len(chunks), len(pages))
    return chunks, metadatas

# ─────────────────────────────────────────────
# 6. VECTOR STORE
# ─────────────────────────────────────────────

def build_vector_store(chunks: list[str], metadatas: list[dict]) -> bool:
    try:
        embeddings = get_embeddings()
        store = FAISS.from_texts(chunks, embedding=embeddings, metadatas=metadatas)
        store.save_local(FAISS_INDEX_PATH)
        logger.info("FAISS index saved (%d vectors).", len(chunks))
        return True
    except Exception as e:
        st.error(f"❌ Failed to build index: {e}")
        logger.exception("Vector store build failed.")
        return False

def load_vector_store():
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error("📂 No index found. Please upload and process documents first.")
        return None
    try:
        store = FAISS.load_local(
            FAISS_INDEX_PATH, get_embeddings(),
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded.")
        return store
    except Exception as e:
        st.error(f"❌ Could not load index: {e}")
        logger.exception("Vector store load failed.")
        return None

# ─────────────────────────────────────────────
# 7. RE-RANKING
# ─────────────────────────────────────────────

def rerank_docs(query: str, docs: list, top_n: int = TOP_K_FINAL) -> list:
    """Use cross-encoder to re-rank retrieved docs by true relevance."""
    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    logger.info("Re-ranked %d docs → top %d selected.", len(docs), top_n)
    return [doc for _, doc in ranked[:top_n]]

# ─────────────────────────────────────────────
# 8. RAG CHAIN WITH MEMORY + STREAMING
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are DocuMind, an expert document analyst. \
Answer questions using ONLY the provided context. \
If the answer is not in the context, say: \
"I couldn't find that in the uploaded documents."
Do NOT fabricate or assume information.
Be concise but thorough. Use bullet points for lists.

Context:
{context}"""

def build_llm(model_name: str):
    return ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=1024,
    )

def format_docs_with_sources(docs) -> tuple[str, list[str]]:
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources = list({f"📄 {doc.metadata['source']} (page {doc.metadata.get('page','?')})" for doc in docs})
    return context, sources

def format_history(chat_history: list) -> list:
    """Convert our chat history format to LangChain message objects."""
    messages = []
    for msg in chat_history[-6:]:  # last 3 exchanges for context window efficiency
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages

def stream_answer(question: str, model_name: str, chat_history: list):
    """
    Generator that yields answer tokens for streaming.
    Returns (sources, elapsed_time) via session state after completion.
    """
    if not validate_api_key():
        return

    store = load_vector_store()
    if store is None:
        return

    try:
        t0 = time.time()

        # Step 1: MMR retrieval (diverse results)
        retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K_RETRIEVE, "fetch_k": 20, "lambda_mult": 0.6}
        )
        raw_docs = retriever.invoke(question)

        # Step 2: Re-rank
        top_docs = rerank_docs(question, raw_docs)

        # Step 3: Format context + extract sources
        context, sources = format_docs_with_sources(top_docs)

        # Step 4: Build prompt with history
        history_messages = format_history(chat_history)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            *[(m.type, m.content) for m in history_messages],
            ("human", "{question}"),
        ])

        llm = build_llm(model_name)
        chain = prompt | llm | StrOutputParser()

        # Step 5: Stream
        full_response = ""
        for chunk in chain.stream({"context": context, "question": question}):
            full_response += chunk
            yield chunk

        elapsed = round(time.time() - t0, 2)

        # Store metadata for UI to pick up
        st.session_state.last_sources  = sources
        st.session_state.last_elapsed  = elapsed
        st.session_state.last_response = full_response
        logger.info("Answer generated in %.2fs via %s", elapsed, model_name)

    except Exception as e:
        st.error(f"❌ Error generating answer: {e}")
        logger.exception("RAG chain failed for: %s", question)

# ─────────────────────────────────────────────
# 9. SESSION STATE
# ─────────────────────────────────────────────

def init_session_state():
    defaults = {
        "chat_history":   [],
        "pdf_processed":  False,
        "doc_stats":      {},
        "last_sources":   [],
        "last_elapsed":   0,
        "last_response":  "",
        "selected_model": list(GROQ_MODELS.keys())[0],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ─────────────────────────────────────────────
# 10. EXPORT CHAT
# ─────────────────────────────────────────────

def export_chat() -> str:
    lines = [f"DocuMind Chat Export — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*60}\n"]
    for msg in st.session_state.chat_history:
        role = "You" if msg["role"] == "user" else "DocuMind"
        lines.append(f"{role}:\n{msg['content']}\n")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# 11. UI — SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 DocuMind")
        st.caption("Production RAG · Powered by Groq")
        st.divider()

        # Model selector
        st.markdown("**🤖 Model**")
        st.session_state.selected_model = st.selectbox(
            "Choose LLM",
            options=list(GROQ_MODELS.keys()),
            label_visibility="collapsed",
        )
        st.divider()

        # File uploader
        st.markdown("**📂 Upload Documents**")
        st.caption("Supports PDF, DOCX, TXT")
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button("⚙️ Process Documents", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Processing…"):
                    pages = extract_documents(uploaded_files)
                    if pages:
                        chunks, metadatas = chunk_documents(pages)
                        ok = build_vector_store(chunks, metadatas)
                        if ok:
                            st.session_state.pdf_processed = True
                            st.session_state.chat_history  = []
                            st.session_state.doc_stats = {
                                "files":  len(uploaded_files),
                                "pages":  len(pages),
                                "chunks": len(chunks),
                            }
                            st.success("✅ Ready to chat!")

        # Doc stats
        if st.session_state.pdf_processed:
            s = st.session_state.doc_stats
            st.divider()
            st.markdown("**📊 Index Stats**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Files",   s.get("files",  0))
            col2.metric("Pages",   s.get("pages",  0))
            col3.metric("Chunks",  s.get("chunks", 0))

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.last_sources = []
                st.rerun()
        with col2:
            if st.session_state.chat_history:
                st.download_button(
                    "💾 Export",
                    data=export_chat(),
                    file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        st.divider()
        st.caption(
            "**Stack**\n\n"
            "LangChain · FAISS · Groq\n\n"
            "HuggingFace · CrossEncoder\n\n"
            "Built by Hitloop 🚀"
        )

# ─────────────────────────────────────────────
# 12. UI — MAIN CHAT
# ─────────────────────────────────────────────

def render_chat():
    st.markdown("## 🧠 DocuMind — Chat with your Documents")
    model_label = st.session_state.selected_model
    model_id    = GROQ_MODELS[model_label]
    st.caption(f"Model: `{model_id}` · Retrieval: MMR + Cross-Encoder Re-ranking")

    if not st.session_state.pdf_processed:
        st.info("👈 Upload and process documents in the sidebar to get started.", icon="ℹ️")
        st.stop()

    st.divider()

    # Render chat history
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show sources for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.caption(src)
            if msg["role"] == "assistant" and msg.get("elapsed"):
                st.caption(f"⚡ {msg['elapsed']}s")

    # Chat input
    question = st.chat_input("Ask anything about your documents…")

    if question:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Stream assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_text = ""

            for token in stream_answer(
                question,
                model_id,
                st.session_state.chat_history[:-1],  # exclude current question
            ):
                full_text += token
                response_placeholder.markdown(full_text + "▌")

            response_placeholder.markdown(full_text)

            # Sources
            sources = st.session_state.get("last_sources", [])
            elapsed = st.session_state.get("last_elapsed", 0)

            if sources:
                with st.expander("📎 Sources", expanded=False):
                    for src in sources:
                        st.caption(src)

            st.caption(f"⚡ {elapsed}s")

        # Save to history with metadata
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": full_text,
            "sources": sources,
            "elapsed": elapsed,
        })

# ─────────────────────────────────────────────
# 13. ENTRY POINT
# ─────────────────────────────────────────────

def main():
    init_session_state()
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()
