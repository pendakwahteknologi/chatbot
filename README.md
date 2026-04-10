# JKST Master AI — RAG Chatbot System

Pembantu Pintar AI untuk Jabatan Kehakiman Syariah Terengganu (JKST).

## 3-Mode RAG System

This system supports 3 operating modes, controlled by `RAG_MODE` in `backend.env` or per-request via the `mode` parameter:

| Mode | `RAG_MODE=` | Description |
|---|---|---|
| **Cloud** | `google` | Google Vertex AI RAG + Gemini + Cohere Rerank |
| **Local** | `local` | ChromaDB + Mesolitica Malay Embeddings + Gemini LLM |
| **Ultra** | `ultra` | Hybrid Search + Query Expansion + Cross-Encoder Rerank + Self-Eval + CoT |

## Frontend Interfaces

| URL | Mode | Theme |
|---|---|---|
| `/` | Cloud (Google) | Blue |
| `/local.html` | Local | Green |
| `/ultra.html` | Ultra | Purple + Gold |
| `/demo.html` | Comparison landing page | Gradient |

## Project Structure

```
project/
├── backend/
│   ├── app.py              # FastAPI backend (main application)
│   └── providers.py        # RAG provider abstraction (3 modes)
├── frontend/
│   ├── index.html          # Cloud mode UI (original)
│   ├── local.html          # Local mode UI (green theme)
│   ├── ultra.html          # Ultra mode UI (purple theme)
│   ├── demo.html           # Comparison landing page
│   ├── architecture.html   # System architecture docs
│   └── changelog.html      # Version changelog
├── knowledge/              # Markdown knowledge files (20 files)
├── configs/
│   ├── backend.env.template    # Environment variables template
│   ├── jkst-master-ai.service  # Systemd service file
│   └── jkst-master-ai.conf     # Nginx config
├── scripts/
│   ├── setup.sh            # Full setup script
│   └── ingest.sh           # Re-ingest documents into ChromaDB
├── requirements.txt        # Python dependencies
└── .gitignore
```

## Quick Setup

```bash
git clone <this-repo> && cd project
chmod +x scripts/*.sh
./scripts/setup.sh
```

## Documents

Place PDF/DOCX/MD/TXT files in `/opt/jkst-master-ai/documents/`, then run:

```bash
./scripts/ingest.sh
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | Chat (accepts `mode` parameter) |
| `/api/chat/stream` | POST | Streaming chat |
| `/api/mode` | GET | Current mode info |
| `/api/health` | GET | Health check |
| `/api/documents` | GET | List documents |
| `/api/feedback` | POST | Submit feedback |
| `/api/voice/transcribe` | POST | Speech-to-text |
| `/api/voice/synthesize` | POST | Text-to-speech |
| `/api/telegram/webhook` | POST | Telegram bot |

## Tech Stack

| Component | Cloud Mode | Local/Ultra Mode |
|---|---|---|
| LLM | Gemini 2.0 Flash | Gemini (swappable to any OpenAI-compatible API) |
| Embeddings | Google (via RAG API) | Mesolitica mistral-embedding-191m-8k (Malay) |
| Vector DB | Google Vertex AI RAG | ChromaDB (local) |
| Reranking | Cohere rerank-v3.5 | Keyword (local) / Cross-encoder (ultra) |
| Web Search | Tavily | Tavily |
| STT | OpenAI Whisper | OpenAI Whisper |
| TTS | Google Cloud TTS | Google Cloud TTS |

## License

Internal use — Jabatan Kehakiman Syariah Terengganu & Pendakwah Teknologi.
