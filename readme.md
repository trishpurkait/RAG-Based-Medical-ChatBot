# 🩺 MediQuery — Production-Style Conversational Medical RAG Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0.3-black?style=flat-square&logo=flask)
![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green?style=flat-square)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-purple?style=flat-square)
![NVIDIA NIM](https://img.shields.io/badge/NVIDIA_NIM-LLaMA_3.1_70B-76B900?style=flat-square&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**A production-grade conversational RAG chatbot for medical Q&A — featuring hybrid retrieval (Dense + BM25), cross-encoder reranking, session-based memory, and a single-LLM-call optimised pipeline.**

</div>

---

## 📸 Screenshots

| Welcome Screen | Chat in Action |
|---|---|
| ![Welcome](welcome.png) | ![Chat](chat.png) |

---

## 📌 Project Overview

MediQuery is an end-to-end conversational Retrieval-Augmented Generation (RAG) system built for the medical domain. It goes significantly beyond basic RAG by implementing a **hybrid retrieval architecture**, a **cross-encoder reranking layer**, **session-based conversational memory**, and an **optimised single-LLM-call pipeline** — engineering decisions that together reduce hallucinations, improve retrieval quality, and enable coherent multi-turn dialogue.

This project is designed as a production-ready GenAI portfolio showcase demonstrating advanced RAG engineering patterns used in real-world AI systems.

---

## ⚡ Basic RAG vs. Production RAG — What Changed

| Capability | Basic RAG | MediQuery (Production RAG) |
|---|---|---|
| Retrieval | Dense vector search only | **Hybrid: Dense (k=8) + BM25 (k=5)** |
| Reranking | None | **Cross-encoder reranker (top\_k=3)** |
| Conversation | Single-turn, stateless | **Multi-turn with session memory** |
| Context injection | None | **Last 3 turns injected into query** |
| LLM calls per query | 2 (rewrite + answer) | **1 (answer only)** |
| Query rewriting | Extra LLM call | **Removed — context injected directly** |
| Hallucination risk | Higher | **Lower — reranked, grounded context** |
| Keyword matching | Weak | **Strong — BM25 handles exact terms** |
| Error resilience | None | **try/except on every module** |
| Source attribution | Basic | **Per-chunk metadata from Pinecone** |

---

## 🏗️ System Architecture

### Full Retrieval & Answer Pipeline

```
User Question  +  Session ID
        │
        ▼
┌──────────────────────────────────┐
│       Memory: get_history()      │  ← fetches last MAX_HISTORY=5 turns
│       format_history() → str     │    formats last 3 turns as context string
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│             hybrid_retrieve(question, history)           │
│                                                          │
│  1. Build augmented_query:                               │
│     "Conversation: {history}\nCurrent Question: {q}"    │
│                                                          │
│  2. Dense Retrieval  ──────────────────────────────────► │
│     Pinecone similarity_search(augmented_query, k=8)     │
│                                                          │
│  3. BM25 Sparse Retrieval  ────────────────────────────► │
│     BM25Retriever.from_documents(dense_docs)             │
│     bm25.invoke(question)  →  k=5 docs                  │
│                                                          │
│  4. Merge + Deduplicate                                  │
│     dense_docs + sparse_docs → unique_docs (by content)  │
│                                                          │
│  5. rerank_documents(question, unique_docs, top_k=3)     │
│     CrossEncoder scores (query, chunk) pairs             │
│     Sorted by descending relevance score                 │
│     Returns top 3 most relevant chunks                   │
└──────────────────────────────────────────────────────────┘
        │
        ▼  final_docs (3 reranked chunks)
┌──────────────────────────────────┐
│       generate_answer()          │
│                                  │
│  context = join(doc.page_content)│
│  chain = prompt | llm            │
│  chain.invoke({context, input})  │  ← single LLM call
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│   update_history(session, q, a)  │  ← stores turn, trims to MAX_HISTORY=5
└──────────────────────────────────┘
        │
        ▼
  JSON → { answer, sources }  →  Browser UI
```

### Memory Flow

```
Session ID
    │
    ▼
chat_sessions[session_id]  →  [ {user, assistant}, ... ]
                                        │
                          format_history() → last 3 turns as string
                                        │
                          injected into augmented_query for retrieval
                          (no extra LLM call — pure string injection)
                                        │
                          update_history() → append + trim to 5 turns
```

---

## 🔬 Key Engineering Decisions

### 1. Hybrid Retrieval: Dense + BM25

Pure dense retrieval struggles with **exact medical terminology** — drug names, dosage values, diagnostic codes. BM25 sparse retrieval handles these precisely.

The pipeline:
- **Dense (Pinecone, k=8):** MiniLM-L6-v2 embeddings (384-dim) capture semantic meaning. "Heart attack" matches "myocardial infarction".
- **BM25 (k=5):** Built dynamically from the dense results using `BM25Retriever.from_documents()`. Handles exact keyword frequency — "metformin 500mg" returns exact matches.
- **Merge + Deduplicate:** Results from both are combined and deduplicated by `page_content` string comparison, ensuring no repeated chunks reach the reranker.

This maximises **recall** (more candidates) while the reranker maximises **precision** (best 3 survive).

### 2. Cross-Encoder Reranking

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Unlike bi-encoders (which score query and document independently), a cross-encoder processes `(query, chunk)` pairs **jointly**, producing true semantic relevance scores. The reranker:

1. Builds pairs: `[[query, chunk.page_content], ...]`
2. Runs `reranker.predict(pairs)` → relevance scores
3. Sorts descending, returns `top_k=3`

**Effect:** Irrelevant chunks retrieved by keyword overlap or semantic approximation are filtered out before reaching the LLM. This directly reduces hallucination and improves answer grounding.

### 3. Single LLM Call — No Query Rewriting

The original pipeline used a separate LLM call to rewrite follow-up questions (`question_chain = rewrite_prompt | llm`), which added latency and API cost.

The updated pipeline **eliminates this** by injecting conversation history directly into the retrieval query as a string:

```python
augmented_query = f"""
Conversation:
{history_text}

Current Question:
{question}
"""
dense_docs = search.similarity_search(augmented_query, k=8)
```

Pinecone's similarity search handles the context-aware retrieval without any extra LLM call. The LLM is called **once**, only for final answer generation.

### 4. Conversation-Aware Memory

`memory.py` maintains a per-session sliding window:
- `get_history(session_id)` — returns session history, initialises if new
- `format_history(history)` — formats **last 3 turns** as a plain string for query augmentation
- `update_history(session_id, q, a)` — appends new turn, trims to `MAX_HISTORY=5`

This keeps the context window small and focused — 3 turns for retrieval, 5 turns stored — balancing coherence with token efficiency.

### 5. Error Resilience

Every `src/` module is wrapped in a `try/except` block. If any module fails to initialise (missing API key, model unavailable, network issue), it prints a diagnostic message instead of crashing the entire app. Flask continues to handle other requests gracefully.

---

## 🗂️ Project Structure

```
RAG-BASED-MEDICAL-CHATBOT/
│
├── Data/                        # Medical PDF files (not pushed to GitHub)
│   └── medicine-book.pdf
│
├── Experiment/
│   └── full_code.ipynb          # Prototyping notebook
│
├── src/                         # Core application modules
│   ├── __init__.py
│   ├── config.py                # API keys, model name, MAX_HISTORY, index_name
│   ├── loader.py                # PDF loading, chunking, embeddings download
│   ├── retriever.py             # hybrid_retrieve(): Dense + BM25 + dedup + rerank
│   ├── reranker.py              # CrossEncoder reranker, rerank_documents()
│   ├── rag_chain.py             # generate_answer(): context builder + LLM call
│   ├── llm.py                   # NVIDIA NIM LLM (LLaMA 3.1 70B)
│   ├── prompt.py                # System prompt template (ChatPromptTemplate)
│   ├── question_chain.py        # Legacy rewrite chain (retained, unused in v2)
│   └── memory.py                # Session chat history management
│
├── static/
│   └── style.css                # Frontend styles
│
├── templates/
│   └── index.html               # Chat UI (Jinja2 template)
│
├── app.py                       # Flask app — /, /chat, /clear
├── store_vectors.py             # One-time indexing script
├── render.yaml                  # Render deployment blueprint
├── Procfile                     # Gunicorn start command
├── runtime.txt                  # Python 3.11.0
├── setup.py                     # Package setup
├── requirements.txt
└── .env                         # API keys (never commit)
```

---

## 🧩 Module Reference

| Module | Role | Key Detail |
|---|---|---|
| `config.py` | Loads `.env`, exports constants | `MAX_HISTORY=5`, `index_name="medical-chatbot"`, wrapped in try/except |
| `loader.py` | PDF ingestion | `DirectoryLoader` + `PyPDFLoader`; chunk size 500, overlap 50 |
| `retriever.py` | **Hybrid retrieval** | Dense k=8 → BM25 k=5 → merge/dedup → rerank top\_k=3 |
| `reranker.py` | Cross-encoder reranking | `ms-marco-MiniLM-L-6-v2`; scores pairs, sorts descending |
| `rag_chain.py` | Answer generation | Builds context string from docs; `prompt \| llm` single call |
| `llm.py` | LLM client | `ChatOpenAI` → NVIDIA NIM base URL, `temperature=0.2` |
| `prompt.py` | Prompt template | System prompt with `{context}` + `{input}`; no hallucination instruction |
| `memory.py` | Session memory | Sliding window; format uses last 3, stores last 5 |
| `question_chain.py` | Legacy (v1) | Rewrite chain — retained but not called in current pipeline |
| `app.py` | Flask routes | Calls `hybrid_retrieve()` then `generate_answer()` per request |
| `store_vectors.py` | One-time setup | Creates Pinecone index (dim=384, cosine, AWS us-east-1) |

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | LLaMA 3.1 70B (NVIDIA NIM) | Answer generation |
| **Embeddings** | `all-MiniLM-L6-v2` (384-dim) | Dense vector encoding |
| **Vector DB** | Pinecone Serverless | Dense retrieval (k=8) |
| **Sparse Retrieval** | BM25 (LangChain) | Exact keyword matching (k=5) |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` | Cross-encoder relevance scoring |
| **RAG Framework** | LangChain 0.2 | Chain composition, loaders, splitters |
| **Backend** | Flask 3.0 + Gunicorn | HTTP server, Jinja2 templates |
| **Frontend** | Vanilla HTML/CSS | DM Sans + DM Serif Display UI |
| **PDF Parsing** | PyPDF via LangChain | Document loading and chunking |
| **Memory** | In-process Python dict | Session-scoped sliding window |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/trishpurkait/RAG-Based-Medical-ChatBot.git
cd RAG-Based-Medical-ChatBot
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your-pinecone-api-key-here
NVIDIA_NIM=your-nvidia-nim-api-key-here
```

| Variable | Where to get it |
|---|---|
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) → API Keys |
| `NVIDIA_NIM` | [build.nvidia.com](https://build.nvidia.com) → Get API Key |

### 5. Add your medical PDFs

```
Data/
└── medicine-book.pdf      ← place your PDFs here
```

### 6. Index documents into Pinecone (run once)

```bash
python store_vectors.py
```

This creates a Pinecone index named `medical-chatbot` (dim=384, cosine, AWS us-east-1) and upserts all PDF chunks. Only run again if you add new documents.

### 7. Start the app

```bash
python app.py
```

Open **http://localhost:5000**

---

## 🔌 API Reference

### `POST /chat`

```json
// Request
{ "message": "What is diabetes?", "session_id": "user-abc123" }

// Response
{ "answer": "Diabetes is a chronic condition...", "sources": ["Data\\medicine-book.pdf"] }
```

### `POST /clear`

```json
// Request
{ "session_id": "user-abc123" }

// Response
{ "status": "History cleared" }
```

### `GET /`

Serves the chat UI.

---

## ☁️ Render Deployment

```bash
# Push deployment files
git add render.yaml Procfile runtime.txt requirements.txt app.py
git commit -m "render deployment config"
git push
```

On [render.com](https://render.com): **New → Web Service → connect repo**

| Setting | Value |
|---|---|
| Build Command | `pip install -r requirements.txt && pip install -e .` |
| Start Command | `gunicorn app:app --workers 2 --timeout 120` |
| Python Version | `3.11.0` |

Add environment variables in the Render dashboard:

| Key | Value |
|---|---|
| `PINECONE_API_KEY` | your key |
| `NVIDIA_NIM` | your key |

> `Data/` PDFs are in `.gitignore` — the app connects to your pre-populated Pinecone index on startup.

---

## 🔧 Configuration Reference

| Parameter | Location | Value | Effect |
|---|---|---|---|
| `MAX_HISTORY` | `config.py` | `5` | Max turns stored per session |
| `format_history` window | `memory.py` | `[-3:]` | Last 3 turns used for query augmentation |
| Dense retrieval k | `retriever.py` | `8` | Pinecone candidates |
| BM25 k | `retriever.py` | `5` | Sparse candidates |
| Reranker top\_k | `reranker.py` | `3` | Final chunks passed to LLM |
| Chunk size | `loader.py` | `500` | Characters per chunk |
| Chunk overlap | `loader.py` | `50` | Overlap between chunks |
| LLM temperature | `llm.py` | `0.2` | Low = factual, consistent answers |

---

## 🔮 Future Improvements

| Feature | Description |
|---|---|
| **Streaming responses** | Stream LLM tokens to UI using `stream=True` + Server-Sent Events |
| **Redis memory** | Replace in-process `chat_sessions` dict with Redis for persistent, scalable sessions |
| **RAG evaluation** | Integrate RAGAS framework to measure faithfulness, answer relevance, and context recall |
| **Hallucination detection** | Add a post-generation verification layer using NLI models |
| **Observability & tracing** | LangSmith or Arize Phoenix integration for retrieval and generation tracing |
| **Docker deployment** | Containerise with `Dockerfile` + `docker-compose.yml` for portable deployment |
| **Authentication** | Session-based auth or API key middleware to protect the `/chat` endpoint |
| **Re-indexing API** | `POST /ingest` endpoint to add new PDFs without restarting the server |
| **Evaluation dashboard** | Admin panel showing retrieval hit rates, latency, and source usage stats |

---

## ⚠️ Important Notes

- **Not a substitute for professional medical advice.** MediQuery is for informational purposes only.
- The LLM is instructed to answer **only from retrieved documents**. If the answer isn't in the indexed PDFs, it says so.
- Chat history is **in-memory** — resets on Flask restart. Use Redis for persistence.
- Never commit `.env`. It is blocked by `.gitignore`.
- `question_chain.py` is retained for reference but is **not called** in the current pipeline.

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 👤 Author

**TRISH** · [github.com/trishpurkait](https://github.com/trishpurkait) · trishpurkat@gmail.com