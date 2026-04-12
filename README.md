# GX10 Multi-Agency RAG Chatbot System

A production AI chatbot system powering multiple Malaysian government agencies on a single codebase. Currently deployed for **JKST** (Jabatan Kehakiman Syariah Terengganu) and **JPA** (Jabatan Perkhidmatan Awam).

This system implements **multiple RAG (Retrieval-Augmented Generation) architectures** on the same codebase, switchable per-request. Each agency gets its own deployment with agency-specific configuration (`agency_config.py`). Deployed on **NVIDIA GX10 (Grace Blackwell)** with GPU-accelerated embeddings, reranking, and local LLM inference.

### Deployment Modes by Agency

| Agency | Modes | Port |
|--------|-------|------|
| JKST | 3 modes: Cloud (Gemini), Local (YTL Ilmu), Ultra (Gemma 4 via Ollama) | 8001 |
| JPA | 4 modes: Local (YTL Ilmu), Ultra (Gemma 4), Extreme Lite (vLLM), Extreme (vLLM) | 8002 |

---

## Table of Contents

- [Overview](#overview)
- [Architecture Comparison](#architecture-comparison)
  - [Mode 1: Cloud RAG (Google)](#mode-1-cloud-rag-google)
  - [Mode 2: Local RAG (YTL Ilmu)](#mode-2-local-rag-ytl-ilmu)
  - [Mode 3: Ultra RAG (100% Local)](#mode-3-ultra-rag-100-local)
- [Side-by-Side Comparison](#side-by-side-comparison)
- [Hardware Profile](#hardware-profile)
- [GX10 Performance Tuning](#gx10-performance-tuning)
- [Project Structure](#project-structure)
- [Setup Guide](#setup-guide)
- [API Reference](#api-reference)
- [Voice Features](#voice-features)
- [Frontend Interfaces](#frontend-interfaces)
- [Document Ingestion Pipeline](#document-ingestion-pipeline)
- [Configuration Reference](#configuration-reference)

---

## Overview

```
User asks: "Apakah prosedur fasakh di Mahkamah Syariah?"
                            |
                    ┌───────┴───────┐
                    │  mode=?       │
                    └───┬───┬───┬───┘
                        │   │   │
              ┌─────────┘   │   └─────────┐
              v             v             v
         ┌─────────┐  ┌─────────┐  ┌──────────┐
         │  CLOUD  │  │  LOCAL  │  │  ULTRA   │
         │ Gemini  │  │YTL Ilmu │  │ Gemma 4  │
         │ Google  │  │ChromaDB │  │ Ollama   │
         └────┬────┘  └────┬────┘  └────┬─────┘
              │            │            │
              v            v            v
         Same FastAPI  Same FastAPI  Same FastAPI
         Same Frontend Same Frontend + Voice UI
         Cloud APIs    Cloud LLM    100% Local
```

All 3 modes are accessible simultaneously via the **Compare page** (`/compare.html`), which sends one query to all 3 modes in parallel and displays responses side-by-side.

---

## Architecture Comparison

### Mode 1: Cloud RAG (Google)

**`mode=google`** — Full Google Cloud Platform pipeline.

| Component | Technology |
|-----------|-----------|
| LLM | Gemini 2.0 Flash (Vertex AI) |
| Retrieval | Google Vertex AI RAG Corpus |
| Embeddings | Google (server-side) |
| Reranking | Cohere rerank-v3.5 / Gemini |
| Web Search | Tavily + Brave (dual provider, auto-fallback) |
| Document Storage | Google Cloud Storage |

**Flow:** Query → Classify → Vertex AI RAG retrieval → Cohere reranking → Local knowledge search → JKST website scraping → Gemini generation → Response

**Strengths:** Best answer quality, document download links, fastest response (~10s)
**Requires:** Google Cloud credentials, internet connection

---

### Mode 2: Local RAG (YTL Ilmu)

**`mode=local`** — Local retrieval with cloud LLM.

| Component | Technology |
|-----------|-----------|
| LLM | YTL Ilmu v2 (free Malaysian AI API) |
| Retrieval | ChromaDB on NVMe (GPU-accelerated) |
| Embeddings | Mesolitica 191M (Malay-optimized, GPU) |
| Reranking | Keyword overlap scoring |
| Web Search | Tavily + Brave (dual provider, if configured) |

**Flow:** Query → ChromaDB vector search (GPU) → Keyword reranking → Local knowledge search → YTL Ilmu generation → Response

**Strengths:** Fast retrieval (~8-15s), Malay-optimized embeddings, free LLM
**Requires:** Internet (for YTL Ilmu API only)

---

### Mode 3: Ultra RAG (100% Local)

**`mode=ultra`** — Maximum capability, fully local, zero cloud dependency.

| Component | Technology |
|-----------|-----------|
| LLM | **Gemma 4 26B MoE via Ollama** (local GPU) |
| Retrieval | ChromaDB + BM25 hybrid (GPU) |
| Embeddings | Mesolitica 191M (Malay, GPU) |
| Reranking | Cross-encoder ms-marco-MiniLM (GPU) |
| Query Expansion | LLM generates 3 query variants |
| Self-Evaluation | LLM scores own answer 1-5 |
| Follow-ups | LLM suggests 3 follow-up questions |
| Conversation Memory | SQLite |
| STT (Listen) | **Faster-Whisper large-v3 (local GPU)** |
| TTS (Talk) | **Facebook MMS-TTS Malay (local GPU)** |

**Flow:** Query → LLM query expansion (3 variants) → Hybrid search (Vector + BM25 + RRF fusion) → Cross-encoder reranking → Chain-of-Thought generation → Self-evaluation + Follow-ups (parallel) → Response

**Strengths:** 100% offline capable, full RAG pipeline, voice input/output, most thorough answers
**Requires:** Nothing — runs entirely on GX10 GPU

---

## Side-by-Side Comparison

| Feature | Mode 1 (Cloud) | Mode 2 (Local) | Mode 3 (Ultra) |
|---------|----------------|----------------|-----------------|
| LLM | Gemini 2.0 Flash | YTL Ilmu v2 | Gemma 4 (Ollama) |
| Retrieval | Vertex AI RAG | ChromaDB (GPU) | ChromaDB + BM25 (GPU) |
| Embeddings | Google (cloud) | Mesolitica 191M (GPU) | Mesolitica 191M (GPU) |
| Reranking | Cohere API | Keyword overlap | Cross-encoder (GPU) |
| Query Expansion | No | No | Yes (3 variants) |
| Hybrid Search | No | No | Yes (Vector + BM25 RRF) |
| Self-Evaluation | No | No | Yes (1-5 score) |
| Follow-ups | No | No | Yes (3 suggestions) |
| Chain-of-Thought | No | No | Yes (Fahami → Analisis → Jawab) |
| Conversation Memory | No | No | Yes (SQLite) |
| Voice Input (STT) | No | No | Yes (Whisper large-v3, GPU) |
| Voice Output (TTS) | No | No | Yes (MMS-TTS Malay, GPU) |
| Web Search | Tavily + Brave | Tavily + Brave | Tavily + Brave |
| Internet Required | Yes (full) | Yes (LLM only) | **No (100% local)** |
| Response Time | ~10s | ~8-15s | ~60-120s |
| Cost per Query | Google + Cohere API | Free (YTL) | Free (local) |

---

## Hardware Profile

Deployed on **NVIDIA GX10 (Grace Blackwell)**:

| Component | Specification |
|-----------|--------------|
| CPU | 20 ARM cores (10x Cortex-X925 @ 3.9GHz + 10x Cortex-A725) |
| RAM | 122GB unified memory (shared CPU+GPU) |
| GPU | NVIDIA GB10, CUDA 13.0 |
| Storage | 916GB NVMe SSD |
| OS | Ubuntu aarch64, Linux 6.17 |

---

## GX10 Performance Tuning

The following optimizations are applied for this hardware:

### Uvicorn (Application Server)
- 8 workers with `uvloop` async event loop
- 200 concurrent connections per worker (1600 total)
- 2048 connection backlog
- 65s keepalive (matches nginx)
- No access logging (reduces I/O)

### GPU Acceleration
- Sentence-transformer embeddings on CUDA (`EMBEDDING_DEVICE=cuda`)
- Cross-encoder reranking on CUDA (`CROSS_ENCODER_DEVICE=cuda`)
- Batch size 256 for embedding (vs default 32)
- Pre-normalized embeddings for faster cosine similarity
- Model warm-up at load time (avoids CUDA kernel compile latency)

### ChromaDB (Vector Database)
- Persisted on NVMe at `/opt/jkst-master-ai/chroma_db`
- HNSW tuned: M=32, construction_ef=200, search_ef=100, 8 threads

### Nginx (Reverse Proxy)
- 64 keepalive connections to backend
- 180s read timeout for Ultra mode
- SSE streaming endpoint with buffering disabled
- Gzip level 4 (good ratio without CPU waste)
- Rate limiting: 20r/s API, 60r/s general

### Ollama (Local LLM)
- Keep model in memory for 30 minutes (`OLLAMA_KEEP_ALIVE=30m`)
- 4 parallel request slots (`OLLAMA_NUM_PARALLEL=4`)
- Gemma 4 26B MoE (~9.6GB, activates only 3.8B per inference)

### systemd Service
- 16GB memory limit (headroom for ML models across 8 workers)
- GPU access via `video`/`render` groups
- 65536 file descriptor limit
- Security hardening (NoNewPrivileges, ProtectSystem=strict)
- Auto-restart on failure

---

## Project Structure

```
├── backend/
│   ├── app.py              # FastAPI application (~2700 lines)
│   │                       #   - API endpoints (chat, stream, voice, telegram, feedback)
│   │                       #   - Query classification engine
│   │                       #   - Google RAG retrieval + Gemini generation
│   │                       #   - Local knowledge search
│   │                       #   - JKST website scraping
│   │                       #   - CSV conversation logging
│   │                       #   - Local voice endpoints (Whisper STT + MMS TTS)
│   ├── agency_config.py    # Agency-specific configuration (name, keywords, contacts)
│   └── providers.py        # RAG provider abstraction (~1100 lines)
│                           #   - Base classes (Retriever, Reranker, Generator)
│                           #   - LocalRetriever (ChromaDB + GPU embeddings)
│                           #   - UltraRetriever (Hybrid Vector + BM25 + RRF)
│                           #   - UltraReranker (Cross-encoder on GPU)
│                           #   - OllamaLocalGenerator (Gemma 4 via Ollama)
│                           #   - OpenAICompatibleGenerator (YTL Ilmu)
│                           #   - GeminiGenerator (Google Cloud)
│                           #   - Document ingestion pipeline
│                           #   - Mode factory with auto-detection
│
├── frontend/
│   ├── index.html          # Hub page — links to all 3 modes
│   ├── cloud.html          # Mode 1: Cloud RAG interface (blue theme)
│   ├── local.html          # Mode 2: Local RAG interface (green theme)
│   ├── ultra.html          # Mode 3: Ultra RAG interface (purple/gold)
│   │                       #   + Voice UI (mic button + speaker button)
│   ├── compare.html        # Side-by-side 3-column comparison page
│   ├── demo.html           # Original demo comparison page
│   ├── architecture.html   # Technical documentation page
│   ├── changelog.html      # Changelog page
│   ├── jkst-logo.png       # JKST logo
│   └── pkns-logo.jpg       # PKNS logo
│
├── knowledge/              # Markdown knowledge files (19 original + 14 scraped)
│   ├── pengenalan.md       # JKST introduction
│   ├── visi-dan-misi.md    # Vision and mission
│   ├── struktur-organisasi.md
│   ├── soalan-*.md         # FAQ files
│   ├── bahagian-*.md       # Department info
│   ├── laman-web-*.md      # Scraped from syariah.terengganu.gov.my
│   └── ...                 # 33 files total → 300 chunks in ChromaDB
│
├── configs/
│   ├── backend.env.template    # Environment config template (GX10-tuned)
│   ├── jkst-master-ai.service  # systemd unit file (GX10-tuned)
│   └── jkst-master-ai.conf     # Nginx reverse proxy config (GX10-tuned)
│
├── scripts/
│   ├── setup.sh            # GX10 automated setup (GPU detection, model download)
│   └── ingest.sh           # GPU-accelerated ChromaDB re-ingestion
│
├── requirements.txt        # Python dependencies (GPU PyTorch, security patches)
└── README.md               # This file
```

---

## Setup Guide

### Prerequisites
- Ubuntu (aarch64 or x86_64)
- Python 3.11+
- NVIDIA GPU with CUDA (optional, falls back to CPU)
- Ollama (for Mode 3 Ultra)

### Quick Setup (GX10)

```bash
git clone https://github.com/pendakwahteknologi/chatbot.git
cd chatbot
sudo ./scripts/setup.sh
```

The setup script auto-detects your GPU and:
1. Creates `/opt/jkst-master-ai/` with all subdirectories
2. Installs GPU PyTorch + all dependencies in a venv
3. Pre-downloads ML models (embedding + cross-encoder)
4. Installs systemd service and nginx config

### Post-Setup

```bash
# 1. Add your API keys
nano /opt/jkst-master-ai/backend.env

# 2. Place Google credentials (for Mode 1)
cp jkst-credentials.json /opt/jkst-master-ai/

# 3. Pull Ollama model (for Mode 3)
ollama pull gemma4

# 4. Ingest documents into ChromaDB
./scripts/ingest.sh

# 5. Start
sudo systemctl start jkst-master-ai

# 6. Verify
curl http://localhost:8001/api/health
```

### Access
| Page | URL |
|------|-----|
| Hub (mode selector) | `http://<ip>/` |
| Ultra Mode (recommended) | `http://<ip>/ultra.html` |
| Local Mode | `http://<ip>/local.html` |
| Cloud Mode | `http://<ip>/cloud.html` |
| Compare (all 3 side-by-side) | `http://<ip>/compare.html` |
| Architecture Docs | `http://<ip>/architecture.html` |
| API Health | `http://<ip>/api/health` |

---

## API Reference

### POST /api/chat
Standard chat endpoint. Supports per-request mode switching.

```json
{
    "messages": [{"role": "user", "content": "Apa itu JKST?"}],
    "mode": "ultra"  // optional: "google", "local", "ultra"
}
```

**Response:**
```json
{
    "reply": "...",
    "retrieval": [...],
    "query_type": "internal",
    "cache_hit": false,
    "mode": "ultra",
    "self_evaluation": {"relevan": 5, "tepat": 5, "lengkap": 4, "purata": 4.7},
    "followup_suggestions": ["...", "...", "..."]
}
```

### POST /api/chat/stream
SSE streaming endpoint. Same request format, returns Server-Sent Events.

### GET /api/health
Returns system status, current mode, feature flags, and configuration.

### POST /api/feedback
Submit user feedback (thumbs up/down + optional comment).

### GET /api/documents
List available documents.

### GET /api/download/{path}
Download a document from Google Cloud Storage.

### Telegram Integration
Each agency deployment runs its own Telegram bot using polling mode (no webhook needed). The bots support conversation history, /clear to reset, and /help for usage info. Configure via `TELEGRAM_BOT_TOKEN` in the environment file.

- **JKST:** @JKST_Chatbot_GX10_Bot
- **JPA:** @JPA_Chatbot_GX10_Bot

---

## Voice Features

Mode 3 (Ultra) includes fully local voice capabilities:

### Speech-to-Text (Listen)
- **Endpoint:** `POST /api/voice/transcribe/local`
- **Model:** Faster-Whisper large-v3 on CUDA
- **Language:** Malay (ms) auto-detected
- **Input:** Base64-encoded audio (WebM from browser mic)

### Text-to-Speech (Talk)
- **Endpoint:** `POST /api/voice/synthesize/local`
- **Model:** Facebook MMS-TTS Malay (`facebook/mms-tts-zlm`) on CUDA
- **Output:** WAV audio, 22050Hz mono 16-bit PCM

### Cloud Voice (Mode 1 & 2, optional)
- **STT:** `POST /api/voice/transcribe` — OpenAI Whisper API (requires OPENAI_API_KEY)
- **TTS:** `POST /api/voice/synthesize` — Google Cloud TTS (uses GCP credentials)

### Ultra Voice UI
The `ultra.html` page includes:
- **Microphone button** — click to start recording, click again to stop and auto-transcribe
- **Speaker button** on each AI response — click to hear the answer in Malay

---

## Frontend Interfaces

| Interface | File | Theme | Features |
|-----------|------|-------|----------|
| Hub | `index.html` | Dark + gradient | Mode selector with feature comparison cards |
| Cloud | `cloud.html` | Blue | Google Cloud RAG interface |
| Local | `local.html` | Green | Local ChromaDB + YTL Ilmu interface |
| Ultra | `ultra.html` | Purple/Gold | Full-featured: voice, self-eval, follow-ups, CoT |
| Compare | `compare.html` | Dark | 3-column side-by-side, fire all modes in parallel |
| Architecture | `architecture.html` | — | Technical documentation |
| Changelog | `changelog.html` | — | Version history |

---

## Document Ingestion Pipeline

The ingestion script (`scripts/ingest.sh`) processes documents into ChromaDB:

1. **Extract** text from PDF (PyMuPDF), DOCX (python-docx), MD, and TXT files
2. **Chunk** into 500-character segments with 100-character overlap
3. **Embed** using Mesolitica 191M on GPU (batch size 256, ~200 chunks/s)
4. **Store** in ChromaDB with HNSW index (cosine similarity, M=32, ef=200)

### Current Knowledge Base
- **33 files** (19 original markdown + 14 scraped from JKST website)
- **300 chunks** in ChromaDB
- **Sources:** FAQ, department info, court procedures, operating hours, addresses, forms, organization structure, news

### Adding Documents
```bash
# Place PDF/DOCX/MD/TXT files in:
cp your-document.pdf /opt/jkst-master-ai/documents/

# Re-ingest (GPU-accelerated, takes ~10s for 300 chunks)
./scripts/ingest.sh

# Restart service
sudo systemctl restart jkst-master-ai
```

---

## Configuration Reference

### Environment Variables (`/opt/jkst-master-ai/backend.env`)

#### Google Cloud (Mode 1)
| Variable | Description |
|----------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON |
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_RAG_LOCATION` | Vertex AI RAG region (e.g., `asia-southeast1`) |
| `GCP_GEMINI_LOCATION` | Gemini API region (e.g., `us-central1`) |
| `GCP_RAG_CORPUS_ID` | Vertex AI RAG corpus ID |
| `GCP_GEMINI_MODEL` | Gemini model (default: `gemini-2.0-flash-001`) |

#### LLM Provider (Mode 2)
| Variable | Description |
|----------|-------------|
| `LOCAL_LLM_BASE_URL` | YTL Ilmu API base URL |
| `LOCAL_LLM_API_KEY` | YTL Ilmu API key |
| `LOCAL_LLM_MODEL` | Model name (default: `ilmu-text-free-v2`) |

#### Ollama (Mode 3)
| Variable | Description |
|----------|-------------|
| `ULTRA_LLM_BASE_URL` | Ollama API URL (default: `http://localhost:11434/v1`) |
| `ULTRA_LLM_MODEL` | Ollama model (default: `gemma4`) |

#### GPU Acceleration
| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_DEVICE` | `cuda` | Device for sentence-transformer embeddings |
| `CROSS_ENCODER_DEVICE` | `cuda` | Device for cross-encoder reranker |
| `EMBEDDING_BATCH_SIZE` | `256` | Batch size for GPU embedding |
| `INGEST_BATCH_SIZE` | `500` | Batch size for ChromaDB ingestion |

#### Voice (Mode 3)
| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large-v3` | Faster-Whisper model size |
| `WHISPER_DEVICE` | `cuda` | Device for Whisper STT |
| `TTS_LOCAL_MODEL` | `facebook/mms-tts-zlm` | HuggingFace TTS model |
| `TTS_LOCAL_DEVICE` | `cuda` | Device for TTS |

#### API Keys
| Variable | Description |
|----------|-------------|
| `TAVILY_API_KEY` | Tavily web search API key |
| `BRAVE_API_KEY` | Brave Search API key (fallback if Tavily unavailable) |
| `COHERE_API_KEY` | Cohere reranking API key (Mode 1) |
| `OPENAI_API_KEY` | OpenAI Whisper STT key (cloud voice) |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |

#### Other
| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_MODE` | `ultra` | Default mode: `google`, `local`, `ultra` |
| `CACHE_TTL_SECONDS` | `600` | Response cache TTL (10 min) |
| `LOCAL_RETRIEVAL_TOP_K` | `15` | Docs to retrieve (local mode) |
| `ULTRA_RETRIEVAL_TOP_K` | `20` | Docs to retrieve (ultra mode) |
| `RAG_INITIAL_RESULTS` | `20` | Initial retrieval count |
| `RAG_FINAL_RESULTS` | `7` | After reranking |
| `HF_HOME` | `/opt/jkst-master-ai/.hf_cache` | HuggingFace model cache |

---

## Credits

- **JKST** — Jabatan Kehakiman Syariah Terengganu
- **JPA** — Jabatan Perkhidmatan Awam
- **Mesolitica** — Malay-optimized embedding model (191M)
- **Google** — Gemini, Vertex AI, Gemma 4
- **YTL AI Labs** — Ilmu text-free-v2 (free Malaysian AI)
- **Meta/Facebook** — MMS-TTS multilingual speech synthesis
- **OpenAI** — Whisper speech recognition architecture
- **Ollama** — Local LLM serving
- **Brave** — Web search API (fallback provider)
- **Tavily** — AI-optimized web search API

---

## License

This project is for internal government agency use and educational purposes.
