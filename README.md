# JKST Master AI — 3-Mode RAG Chatbot System

Pembantu Pintar AI untuk **Jabatan Kehakiman Syariah Terengganu (JKST)** — a production AI chatbot serving the Terengganu Shariah Judiciary Department in Malaysia.

This system demonstrates **3 different RAG (Retrieval-Augmented Generation) architectures** running on the same codebase, switchable via a single environment variable or per-request parameter. Built for production use and as a teaching tool for RAG comparison classes.

---

## Table of Contents

- [Overview](#overview)
- [Architecture Comparison](#architecture-comparison)
  - [Mode 1: Cloud RAG (Google)](#mode-1-cloud-rag-google)
  - [Mode 2: Local RAG (On-Premise)](#mode-2-local-rag-on-premise)
  - [Mode 3: Ultra RAG (Maximum Capability)](#mode-3-ultra-rag-maximum-capability)
- [Side-by-Side Comparison](#side-by-side-comparison)
- [How Mode Switching Works](#how-mode-switching-works)
- [Project Structure](#project-structure)
- [Setup Guide](#setup-guide)
- [API Reference](#api-reference)
- [Frontend Interfaces](#frontend-interfaces)
- [Document Ingestion Pipeline](#document-ingestion-pipeline)
- [Configuration Reference](#configuration-reference)
- [Performance Benchmarks](#performance-benchmarks)

---

## Overview

```
User asks: "Apakah prosedur fasakh di Mahkamah Syariah?"
                            |
                    ┌───────┴───────┐
                    │   RAG_MODE?   │
                    └───┬───┬───┬───┘
                        │   │   │
              ┌─────────┘   │   └─────────┐
              v             v             v
         ┌─────────┐  ┌─────────┐  ┌─────────┐
         │  CLOUD  │  │  LOCAL  │  │  ULTRA  │
         │ Google  │  │ChromaDB │  │ Hybrid  │
         └────┬────┘  └────┬────┘  └────┬────┘
              │            │            │
              v            v            v
         Same FastAPI  Same FastAPI  Same FastAPI
         Same Frontend Same Frontend Same Frontend
         Same Answer   Same Answer   Same Answer
         Format        Format        Format
```

All 3 modes share:
- The same FastAPI backend (`app.py`)
- The same frontend interfaces
- The same API endpoints and response format
- The same query classification logic
- The same logging and caching system

What differs is **how documents are retrieved, ranked, and how the LLM generates answers**.

---

## Architecture Comparison

### Mode 1: Cloud RAG (Google)

**`RAG_MODE=google`** — Production mode using Google Cloud Platform services.

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLOUD MODE PIPELINE                       │
└──────────────────────────────────────────────────────────────────┘

User Question
     │
     ▼
┌─────────────────┐
│ Query Classifier │  ← Pure Python keyword matching
│ (local, no API)  │    Classifies: internal / external / jkst_news / hybrid
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL (Multi-Source)                       │
│                                                                   │
│  ┌──────────────────────┐  ┌────────────────────┐                │
│  │  Google Vertex AI RAG │  │  Local Knowledge   │                │
│  │  (PRIMARY SOURCE)     │  │  (SECONDARY)       │                │
│  │                       │  │                    │                │
│  │  • 500+ PDF/DOCX docs │  │  • 20 markdown     │                │
│  │  • Google embeddings  │  │    files           │                │
│  │  • Semantic search    │  │  • Keyword match   │                │
│  │  • Returns top 15     │  │  • Returns top 5   │                │
│  └───────────┬───────────┘  └────────┬───────────┘                │
│              │                       │                            │
│  ┌───────────┴───────────┐  ┌────────┴───────────┐                │
│  │  Tavily Web Search    │  │  JKST Website      │                │
│  │  (SUPPLEMENTARY)      │  │  Scraper           │                │
│  │  • External queries   │  │  • Live news       │                │
│  │  • Fallback source    │  │  • Official pages  │                │
│  └───────────┬───────────┘  └────────┬───────────┘                │
│              └──────────┬────────────┘                            │
└─────────────────────────┼────────────────────────────────────────┘
                          │
                          ▼
               ┌──────────────────┐
               │  Cohere Reranking │  ← Cross-encoder API (rerank-v3.5)
               │  15 docs → top 5  │    Scores each doc against query
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  Gemini 2.0 Flash │  ← Google LLM via Vertex AI
               │  (Generation)     │    Temperature: 0.2
               │                   │    Max tokens: 2000
               │  System prompt:   │    Malay language optimized
               │  JKST-specific    │    Source priority rules
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  Response + Refs  │  ← Includes document download links
               │  + Source list    │    from Google Cloud Storage
               └──────────────────┘
```

**Key characteristics:**
- **Retrieval**: Google Vertex AI RAG API — documents are pre-indexed in a cloud corpus with Google's proprietary embeddings
- **Embeddings**: Handled by Google (server-side, not visible to us) — high quality multilingual
- **Reranking**: Cohere rerank-v3.5 API — specialized cross-encoder model for relevance scoring
- **Generation**: Gemini 2.0 Flash via Vertex AI — fast, high quality, Malay-capable
- **Document storage**: Google Cloud Storage (GCS) bucket — provides download links in responses
- **Strengths**: Best answer quality, fastest response time (~10s), document download links
- **Weaknesses**: Requires internet + Google Cloud account + API keys, costs money per query

**External services used:**
1. Google Vertex AI RAG API (retrieval)
2. Google Gemini API (generation)
3. Cohere API (reranking)
4. Google Cloud Storage (documents)
5. Tavily API (web search)

---

### Mode 2: Local RAG (On-Premise)

**`RAG_MODE=local`** — Everything runs locally except the LLM (uses Gemini temporarily, swappable to any OpenAI-compatible API).

```
┌──────────────────────────────────────────────────────────────────┐
│                        LOCAL MODE PIPELINE                        │
└──────────────────────────────────────────────────────────────────┘

User Question
     │
     ▼
┌─────────────────┐
│ Query Classifier │  ← Same as Cloud mode (pure Python)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL (Local Only)                         │
│                                                                   │
│  ┌──────────────────────────────────────┐                        │
│  │         ChromaDB Vector Search        │                        │
│  │         (PRIMARY SOURCE)              │                        │
│  │                                       │                        │
│  │  Embedding Model:                     │                        │
│  │  mesolitica/mistral-embedding-191m    │                        │
│  │  • 191M parameters                   │                        │
│  │  • 768-dimensional vectors           │                        │
│  │  • Trained on Malay text             │                        │
│  │  • 8K token context window           │                        │
│  │                                       │                        │
│  │  Process:                             │                        │
│  │  1. Encode query → 768-dim vector    │                        │
│  │  2. Cosine similarity search         │                        │
│  │  3. Return top 10 chunks             │                        │
│  └──────────────────┬───────────────────┘                        │
│                     │                                             │
│  ┌──────────────────┴───────────────────┐                        │
│  │     Local Knowledge (Keyword Search)  │                        │
│  │     • 20 markdown files              │                        │
│  │     • TF-IDF style scoring           │                        │
│  │     • Filename + content matching    │                        │
│  └──────────────────┬───────────────────┘                        │
└─────────────────────┼────────────────────────────────────────────┘
                      │
                      ▼
               ┌──────────────────┐
               │ Simple Reranking  │  ← Local keyword overlap scoring
               │ (No external API) │    Combines vector score + keyword match
               │ 10 docs → top 5   │    No cross-encoder, just heuristics
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  Gemini 2.0 Flash │  ← Same LLM (temporary)
               │  (via providers)  │    Swappable to: YTL Ilmu, Ollama,
               │                   │    any OpenAI-compatible API
               │  Same prompt as   │
               │  Cloud mode       │
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  Response + Refs  │  ← Sources show "ChromaDB" origin
               │  (no download     │    No GCS download links
               │   links)          │
               └──────────────────┘
```

**Key characteristics:**
- **Retrieval**: ChromaDB (local SQLite-backed vector database) — documents are chunked and embedded locally
- **Embeddings**: Mesolitica mistral-embedding-191m-8k — a Malay-native embedding model from Malaysian AI lab
- **Reranking**: Simple keyword overlap heuristic — no external API, combines vector similarity + keyword matching
- **Generation**: Gemini 2.0 Flash (temporary) — designed to be swapped to local LLM or any OpenAI-compatible API
- **Document storage**: Local filesystem (`/opt/jkst-master-ai/documents/`)
- **Strengths**: No cloud dependency for RAG (only LLM), all data stays on-premise, no per-query retrieval cost
- **Weaknesses**: Lower retrieval quality (smaller embedding model), basic reranking, slower first load (model loading)

**External services used:**
1. Gemini API (generation only — swappable)
2. Tavily API (web search — optional)

**What runs locally:**
- ChromaDB vector database
- Mesolitica embedding model (191M params, CPU)
- Query classification
- Document chunking and ingestion
- Keyword-based reranking

**How document ingestion works:**
```
PDF/DOCX/MD/TXT files
        │
        ▼
┌─────────────────┐
│  Text Extraction │  PyMuPDF (PDF) + python-docx (DOCX)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking      │  500 chars per chunk, 100 char overlap
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding      │  mesolitica/mistral-embedding-191m-8k
│   (768-dim)      │  Runs on CPU (~30 min for 12K chunks)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Store in        │  ChromaDB with cosine similarity
│  ChromaDB        │  Persistent on disk
└─────────────────┘
```

---

### Mode 3: Ultra RAG (Maximum Capability)

**`RAG_MODE=ultra`** — All smart RAG techniques enabled. Demonstrates what state-of-the-art RAG looks like.

```
┌──────────────────────────────────────────────────────────────────┐
│                        ULTRA MODE PIPELINE                        │
└──────────────────────────────────────────────────────────────────┘

User Question: "Apakah prosedur fasakh?"
     │
     ▼
┌─────────────────┐
│ Query Classifier │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 1: QUERY EXPANSION                         │
│                                                                   │
│  LLM generates 3 search variants of the original query:          │
│                                                                   │
│  Original: "Apakah prosedur fasakh?"                             │
│  Variant 1: "Tatacara pembubaran perkahwinan di Mahkamah"       │
│  Variant 2: "Proses permohonan fasakh di Mahkamah Syariah"      │
│  Variant 3: "Langkah-langkah perceraian menurut hukum syarak"   │
│                                                                   │
│  Purpose: Different phrasings catch different relevant documents  │
│  Cost: 1 additional LLM call (~2 seconds)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ (4 queries total)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               STEP 2: HYBRID RETRIEVAL                           │
│                                                                   │
│  Two search methods run in parallel, then results are fused:     │
│                                                                   │
│  ┌─────────────────────┐    ┌─────────────────────┐              │
│  │   VECTOR SEARCH     │    │   BM25 KEYWORD      │              │
│  │   (Semantic)        │    │   SEARCH (Lexical)  │              │
│  │                     │    │                     │              │
│  │ • All 4 queries     │    │ • Original query    │              │
│  │   searched against  │    │   tokenized         │              │
│  │   ChromaDB          │    │ • BM25Okapi scoring │              │
│  │ • Cosine similarity │    │ • Term frequency    │              │
│  │ • Best score per    │    │   matching          │              │
│  │   document kept     │    │ • Catches exact     │              │
│  │ • Finds: semantically│   │   terms like        │              │
│  │   similar content   │    │   "fasakh", "MS 27" │              │
│  └──────────┬──────────┘    └──────────┬──────────┘              │
│             │                          │                          │
│             └────────┬─────────────────┘                          │
│                      ▼                                            │
│  ┌───────────────────────────────────────┐                       │
│  │   RECIPROCAL RANK FUSION (RRF)        │                       │
│  │                                        │                       │
│  │   RRF Score = 1/(k + rank_vector)      │                       │
│  │             + 1/(k + rank_bm25)        │                       │
│  │   where k = 60                         │                       │
│  │                                        │                       │
│  │   Documents found by BOTH methods      │                       │
│  │   get boosted scores (more reliable)   │                       │
│  │                                        │                       │
│  │   Result: 15 best fused documents      │                       │
│  └───────────────────┬───────────────────┘                       │
└──────────────────────┼───────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│               STEP 3: CROSS-ENCODER RERANKING                    │
│                                                                   │
│  Model: cross-encoder/ms-marco-MiniLM-L-6-v2                    │
│                                                                   │
│  Unlike vector search (bi-encoder, encodes query and doc          │
│  separately), cross-encoder processes query+doc TOGETHER:         │
│                                                                   │
│  Input:  [query] [SEP] [document_text]                           │
│  Output: relevance score (0.0 to 5.0+)                           │
│                                                                   │
│  This gives much more accurate relevance judgments because        │
│  it sees the full interaction between query and document.         │
│                                                                   │
│  15 documents → scored → sorted → top 5 selected                 │
│                                                                   │
│  Example scores:                                                  │
│    #1: soalan-lazim-umum.md         (score: 3.1493)              │
│    #2: pengenalan.md                (score: 1.6231)              │
│    #3: khidmat-nasihat.md           (score: 1.6230)              │
│    #4: bahagian-kehakiman.md        (score: 0.8941)              │
│    #5: unit-sulh.md                 (score: 0.5122)              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│            STEP 4: CHAIN-OF-THOUGHT GENERATION                   │
│                                                                   │
│  Ultra mode uses a structured thinking prompt:                    │
│                                                                   │
│  1. FAHAMI  — What is the user actually asking?                  │
│              Identify the true intent behind the question         │
│                                                                   │
│  2. ANALISIS — Review all available documents/contexts            │
│               Identify which sources are relevant                 │
│                                                                   │
│  3. HUBUNGKAN — Connect information from multiple sources         │
│                 Synthesize a complete answer                      │
│                                                                   │
│  4. JAWAB — Provide the structured, complete answer               │
│             Use headings, numbered steps, bullet points           │
│                                                                   │
│  5. SUMBER — Cite sources clearly                                │
│              Flag uncertainty where applicable                    │
│                                                                   │
│  LLM: Gemini 2.0 Flash (swappable)                              │
│  Temperature: 0.2 | Max tokens: 2000                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│            STEP 5: SELF-EVALUATION                               │
│                                                                   │
│  After generating the answer, the LLM evaluates its own          │
│  response quality on 3 criteria (1-5 scale):                     │
│                                                                   │
│  • RELEVAN  — Does the answer address the question?              │
│  • TEPAT    — Is the answer based on provided documents?         │
│  • LENGKAP  — Is the answer sufficiently complete?               │
│                                                                   │
│  Example output:                                                  │
│  {                                                                │
│    "relevan": 3,                                                  │
│    "tepat": 1,                                                    │
│    "lengkap": 2,                                                  │
│    "purata": 2.0,                                                 │
│    "nota": "Dokumen tidak mengandungi maklumat fasakh"           │
│  }                                                                │
│                                                                   │
│  This enables: automatic quality monitoring, retry logic,         │
│  and honest "I don't know" responses.                            │
│                                                                   │
│  Cost: 1 additional LLM call (~2 seconds)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│            STEP 6: FOLLOW-UP SUGGESTIONS                         │
│                                                                   │
│  LLM generates 3 relevant follow-up questions based on           │
│  the conversation:                                                │
│                                                                   │
│  Example:                                                         │
│  → "Apakah alasan yang diterima untuk memfailkan fasakh?"        │
│  → "Di mana boleh dapatkan borang permohonan fasakh?"            │
│  → "Adakah bantuan guaman percuma untuk kes fasakh?"             │
│                                                                   │
│  These appear as clickable buttons in the Ultra UI.              │
│                                                                   │
│  Cost: 1 additional LLM call (~2 seconds)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│            STEP 7: CONVERSATION MEMORY                           │
│                                                                   │
│  SQLite database stores conversation turns:                       │
│                                                                   │
│  conversations table:                                             │
│  | session_id | role      | content          | timestamp        | │
│  |------------|-----------|------------------|------------------| │
│  | abc123     | user      | Prosedur fasakh? | 2026-04-10 18:00 | │
│  | abc123     | assistant | Untuk memfailkan | 2026-04-10 18:00 | │
│                                                                   │
│  Enables: multi-turn context, user history, analytics            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
               ┌──────────────────┐
               │  Final Response   │
               │  • Structured CoT │
               │  • Self-eval score│
               │  • Follow-ups     │
               │  • Source refs     │
               └──────────────────┘
```

**Key characteristics:**
- **Query Expansion**: LLM generates 3 alternative phrasings for better recall
- **Hybrid Search**: Vector (semantic) + BM25 (keyword) combined with Reciprocal Rank Fusion
- **Reranking**: Cross-encoder model (ms-marco-MiniLM-L-6-v2) for accurate relevance scoring
- **Generation**: Chain-of-Thought structured prompt (Fahami → Analisis → Hubungkan → Jawab → Sumber)
- **Self-Evaluation**: AI scores its own answer quality (1-5) on relevance, accuracy, completeness
- **Follow-up Suggestions**: 3 contextual follow-up questions generated
- **Conversation Memory**: SQLite-based persistent memory across turns
- **Strengths**: Most thorough retrieval, honest self-assessment, structured answers, follow-up suggestions
- **Weaknesses**: Slowest (~30s, 4 LLM calls total), highest compute cost

**Total LLM calls per query: 4**
1. Query expansion (generate 3 variants)
2. Answer generation (with Chain-of-Thought)
3. Self-evaluation (score 1-5)
4. Follow-up suggestions (generate 3 questions)

---

## Side-by-Side Comparison

### Feature Matrix

| Feature | Cloud | Local | Ultra |
|---|:---:|:---:|:---:|
| **Retrieval** | Google Vertex AI RAG | ChromaDB vector | ChromaDB + BM25 hybrid |
| **Embeddings** | Google (proprietary) | Mesolitica 191M (Malay) | Mesolitica 191M (Malay) |
| **Embedding Dimensions** | Unknown (Google) | 768 | 768 |
| **Reranking** | Cohere rerank-v3.5 | Keyword overlap | Cross-encoder |
| **LLM** | Gemini 2.0 Flash | Gemini (swappable) | Gemini (swappable) |
| **Query Expansion** | No | No | Yes (3 LLM variants) |
| **Hybrid Search** | No | No | Yes (Vector + BM25 + RRF) |
| **Chain-of-Thought** | No | No | Yes |
| **Self-Evaluation** | No | No | Yes (1-5 score) |
| **Follow-up Suggestions** | No | No | Yes (3 questions) |
| **Conversation Memory** | No | No | Yes (SQLite) |
| **Document Downloads** | Yes (GCS links) | No | No |
| **Response Time** | ~10s | ~30s | ~30s |
| **LLM Calls per Query** | 1 | 1 | 4 |
| **Cloud Dependency** | Full | LLM only | LLM only |
| **Cost per Query** | $$$ | $ | $$ |

### Benchmark Results (Same Query)

**Query**: "Apakah prosedur untuk memfailkan kes fasakh di Mahkamah Syariah Terengganu?"

| Metric | Cloud | Local | Ultra |
|---|---|---|---|
| Response time | **9.7s** | 29.5s | 31.4s |
| Reply length | 2,712 chars | 1,062 chars | **3,761 chars** |
| Sources found | 10 | 10 | 10 |
| Has specific steps | Yes | No | Yes |
| Has download links | Yes | No | No |
| Has Chain-of-Thought | No | No | **Yes** |
| Self-eval score | N/A | N/A | **2.0/5** |
| Follow-up suggestions | N/A | N/A | **3 questions** |
| Answer quality | **Best** | Basic | Structured |

### When to Use Each Mode

| Scenario | Recommended Mode |
|---|---|
| Production deployment | **Cloud** — best quality, fastest |
| Air-gapped / no internet | **Local** — only needs LLM API |
| Teaching RAG concepts | **All 3** — compare side by side |
| Quality-critical answers | **Ultra** — self-eval catches bad answers |
| Cost-sensitive deployment | **Local** — minimal API calls |
| Research / experimentation | **Ultra** — most observable pipeline |

---

## How Mode Switching Works

### Global Mode (Environment Variable)

Set in `backend.env`:
```bash
RAG_MODE=google   # Cloud mode (default)
RAG_MODE=local    # Local mode
RAG_MODE=ultra    # Ultra mode
```

Then restart: `sudo systemctl restart jkst-master-ai`

### Per-Request Mode (API Parameter)

All 3 modes run simultaneously — switch per request:

```bash
# Cloud mode
curl -X POST /api/chat \
  -d '{"messages":[{"role":"user","content":"..."}], "mode":"google"}'

# Local mode
curl -X POST /api/chat \
  -d '{"messages":[{"role":"user","content":"..."}], "mode":"local"}'

# Ultra mode
curl -X POST /api/chat \
  -d '{"messages":[{"role":"user","content":"..."}], "mode":"ultra"}'
```

This is how the 3 frontend interfaces work — each sends a different `mode` parameter to the same backend.

### Provider Architecture

```python
# providers.py — simplified view

class LocalRetriever:
    """ChromaDB vector search"""
    async def retrieve(query) -> (contexts, sources)

class UltraRetriever:
    """Hybrid search with query expansion"""
    def expand_query(query) -> [query, variant1, variant2, variant3]
    async def retrieve(query) -> (contexts, sources)  # vector + BM25 + RRF

class LocalReranker:
    """Keyword overlap scoring"""
    def rerank(query, docs) -> top_5_docs

class UltraReranker:
    """Cross-encoder neural reranking"""
    def rerank(query, docs) -> top_5_docs

class GeminiGenerator:
    """Google Gemini LLM"""
    def generate(prompt) -> response

class OpenAICompatibleGenerator:
    """Any OpenAI-compatible API (YTL Ilmu, Ollama, etc.)"""
    def generate(prompt) -> response

# Factory: get_providers() returns the right set based on RAG_MODE
```

---

## Project Structure

```
project/
├── backend/
│   ├── app.py                 # FastAPI backend (2700+ lines)
│   │                          # - Query classification
│   │                          # - Google RAG retrieval
│   │                          # - Web search (Tavily)
│   │                          # - JKST website scraper
│   │                          # - Gemini generation
│   │                          # - Streaming support
│   │                          # - Telegram bot integration
│   │                          # - Voice (STT/TTS)
│   │                          # - Feedback system
│   │                          # - CSV logging
│   │                          # - Mode routing (google/local/ultra)
│   │
│   └── providers.py           # RAG provider abstraction (540+ lines)
│                               # - LocalRetriever (ChromaDB)
│                               # - UltraRetriever (Hybrid + Query Expansion)
│                               # - LocalReranker (keyword overlap)
│                               # - UltraReranker (cross-encoder)
│                               # - GeminiGenerator
│                               # - OpenAICompatibleGenerator
│                               # - UltraEnhancer (self-eval, followups, memory)
│                               # - Document ingestion (PDF, DOCX, MD, TXT)
│                               # - Mode factory
│
├── frontend/
│   ├── index.html             # Cloud mode UI (blue theme, original)
│   ├── cloud.html             # Cloud mode UI (with mode nav)
│   ├── local.html             # Local mode UI (green theme)
│   ├── ultra.html             # Ultra mode UI (purple+gold, self-eval display)
│   ├── demo.html              # Comparison landing page (3 cards)
│   ├── architecture.html      # System architecture documentation
│   ├── changelog.html         # Version history and roadmap
│   ├── jkst-logo.png          # JKST logo
│   └── pkns-logo.jpg          # PKNS logo
│
├── knowledge/                 # 20 Malay knowledge files
│   ├── pengenalan.md          # JKST introduction
│   ├── visi-dan-misi.md       # Vision, mission, objectives
│   ├── struktur-organisasi.md # Organizational structure
│   ├── bahagian-kehakiman.md  # Judiciary division
│   ├── unit-sulh.md           # Sulh (mediation) unit
│   ├── soalan-lazim-umum.md   # General FAQ
│   ├── soalan-mahkamah-*.md   # Court-specific FAQ
│   └── ...                    # Other department info
│
├── configs/
│   ├── backend.env.template   # Environment variables (no secrets)
│   ├── jkst-master-ai.service # Systemd service file
│   └── jkst-master-ai.conf   # Nginx reverse proxy config
│
├── scripts/
│   ├── setup.sh               # Full setup (venv, deps, configs)
│   └── ingest.sh              # Re-ingest documents into ChromaDB
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excludes: env, documents, chroma_db, venv
└── README.md                  # This file
```

**Not in git (too large / sensitive):**
```
/opt/jkst-master-ai/
├── documents/          # 158 PDF/DOCX files (212 MB) — download from GCS
├── chroma_db/          # ChromaDB vector database (rebuilt by ingest.sh)
├── venv/               # Python virtual environment (rebuilt by setup.sh)
├── logs/               # Conversation logs, feedback CSVs
├── backend.env         # Actual API keys (created from template)
└── memory.db           # SQLite conversation memory (ultra mode)
```

---

## Setup Guide

### Prerequisites

- Ubuntu 22.04+ (or DGX OS for GPU)
- Python 3.11+
- nginx (for reverse proxy)
- 8GB+ RAM (CPU) or GPU with 4GB+ VRAM

### Quick Setup

```bash
git clone https://github.com/pendakwahteknologi/chatbot.git
cd chatbot
chmod +x scripts/*.sh
./scripts/setup.sh
```

### Manual Setup

```bash
# 1. Create project directory
sudo mkdir -p /opt/jkst-master-ai/{logs,knowledge,documents,chroma_db}
sudo chown -R $USER:$USER /opt/jkst-master-ai

# 2. Copy files
cp backend/*.py /opt/jkst-master-ai/
cp knowledge/* /opt/jkst-master-ai/knowledge/

# 3. Create virtual environment
python3 -m venv /opt/jkst-master-ai/venv
source /opt/jkst-master-ai/venv/bin/activate

# 4. Install dependencies (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn httpx pydantic
pip install chromadb sentence-transformers rank-bm25
pip install pymupdf python-docx
pip install google-genai google-cloud-aiplatform google-cloud-storage
pip install openai cohere tavily-python

# 5. Configure environment
cp configs/backend.env.template /opt/jkst-master-ai/backend.env
# Edit backend.env and add your API keys

# 6. Add documents and ingest
# Copy your PDF/DOCX files to /opt/jkst-master-ai/documents/
./scripts/ingest.sh

# 7. Start
sudo cp configs/jkst-master-ai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now jkst-master-ai
```

---

## API Reference

### POST /api/chat

Main chat endpoint. Supports per-request mode switching.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Apakah prosedur fasakh?"}
  ],
  "mode": "ultra"  // optional: "google", "local", "ultra"
}
```

**Response:**
```json
{
  "reply": "## Prosedur Fasakh\n\n1. Sediakan dokumen...",
  "retrieval": [
    {
      "type": "Dokumen (Hybrid Search)",
      "filename": "fasakh-procedure.pdf",
      "page_content": "...",
      "score": 0.0267,
      "vector_score": 0.6143,
      "bm25_score": 5.471,
      "priority": "PRIMARY"
    }
  ],
  "query_type": "internal",
  "cache_hit": false,
  "mode": "ultra",
  "self_evaluation": {
    "relevan": 4,
    "tepat": 3,
    "lengkap": 4,
    "purata": 3.7,
    "nota": "Jawapan cukup lengkap berdasarkan dokumen yang ada"
  },
  "followup_suggestions": [
    "Apakah dokumen yang diperlukan untuk permohonan fasakh?",
    "Berapa lama tempoh proses kes fasakh?",
    "Adakah bantuan guaman untuk kes fasakh?"
  ]
}
```

### GET /api/mode

Returns current mode configuration and feature flags.

### GET /api/health

Returns system health, mode, and feature status.

---

## Frontend Interfaces

| Interface | URL | Color Theme | Mode Sent |
|---|---|---|---|
| Cloud (original) | `/` | Blue (#0061EB) | `google` |
| Local | `/local.html` | Green (#16a34a) | `local` |
| Ultra | `/ultra.html` | Purple+Gold (#9333ea + #f59e0b) | `ultra` |
| Demo Landing | `/demo.html` | Gradient | Links to all 3 |

The Ultra interface has additional UI elements:
- **Self-evaluation display**: Shows relevance/accuracy/completeness scores with progress bars
- **Follow-up suggestion buttons**: Clickable buttons that auto-fill and send the suggested question

---

## Document Ingestion Pipeline

Supports: **PDF**, **DOCX**, **MD**, **TXT**

```bash
# Place files in the documents folder
cp your-documents/*.pdf /opt/jkst-master-ai/documents/

# Run ingestion (clears old data and re-embeds everything)
./scripts/ingest.sh
```

### Pipeline Details

1. **Text Extraction**
   - PDF: PyMuPDF (`fitz`) — extracts text from all pages
   - DOCX: python-docx — extracts paragraphs + table text
   - MD/TXT: Direct file read (UTF-8)

2. **Chunking**
   - Chunk size: 500 characters
   - Overlap: 100 characters
   - Minimum chunk: 50 characters

3. **Embedding**
   - Model: `mesolitica/mistral-embedding-191m-8k-contrastive`
   - Dimensions: 768
   - Context: 8,192 tokens
   - Language: Malay-native (trained by Malaysian AI lab Mesolitica)

4. **Storage**
   - Database: ChromaDB (persistent, SQLite-backed)
   - Similarity: Cosine distance
   - Location: `/opt/jkst-master-ai/chroma_db/`

### Current Statistics
- 158 documents (86 PDF + 71 DOCX + 1 PNG)
- 20 knowledge files (markdown)
- ~11,900 chunks total
- ~71 MB ChromaDB on disk

---

## Configuration Reference

### Environment Variables (`backend.env`)

```bash
# === RAG Mode ===
RAG_MODE=google                    # google | local | ultra

# === Google Cloud (required for Cloud mode) ===
GCP_PROJECT_ID=your-project
GCP_RAG_LOCATION=asia-southeast1
GCP_GEMINI_LOCATION=us-central1
GCP_RAG_CORPUS_ID=your-corpus-id
GCP_GEMINI_MODEL=gemini-2.0-flash-001
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GCS_BUCKET_NAME=your-bucket

# === Reranking ===
ENABLE_RERANKING=true
RERANKER_TYPE=cohere               # cohere | gemini
COHERE_API_KEY=your-key

# === Web Search ===
TAVILY_API_KEY=your-key

# === Voice ===
OPENAI_API_KEY=your-key            # For Whisper STT
TTS_LANGUAGE_CODE=ms-MY
TTS_VOICE_NAME=ms-MY-Wavenet-A

# === Telegram Bot ===
TELEGRAM_BOT_TOKEN=your-token
TELEGRAM_WEBHOOK_URL=https://your-domain.com
TELEGRAM_WEBHOOK_SECRET=your-secret

# === Local/Ultra Mode ===
LOCAL_EMBEDDING_MODEL=mesolitica/mistral-embedding-191m-8k-contrastive
LOCAL_KNOWLEDGE_PATH=/opt/jkst-master-ai/knowledge
LOCAL_DOCUMENTS_PATH=/opt/jkst-master-ai/documents
CHROMA_PERSIST_DIR=/opt/jkst-master-ai/chroma_db
HF_HOME=/opt/jkst-master-ai/logs/.hf_cache

# === Swappable LLM (for local/ultra when not using Gemini) ===
# LOCAL_LLM_BASE_URL=https://api.ytlailabs.tech/v1
# LOCAL_LLM_API_KEY=your-key
# LOCAL_LLM_MODEL=ilmu-text-free-v2

# === General ===
BASE_URL=https://your-domain.com
CSV_LOG_PATH=/opt/jkst-master-ai/logs/conversations.csv
FEEDBACK_CSV_PATH=/opt/jkst-master-ai/logs/feedback.csv
CACHE_TTL_SECONDS=300
```

---

## Performance Benchmarks

Tested on: Ubuntu 24.04, Intel Xeon E5-2620 v4 (4 cores), 7.7 GB RAM, no GPU.

| Metric | Cloud | Local | Ultra |
|---|---|---|---|
| First request (cold) | ~12s | ~35s (model loading) | ~40s (2 models loading) |
| Subsequent requests | ~10s | ~8s | ~30s |
| Memory usage | ~200 MB | ~900 MB | ~1.2 GB |
| ChromaDB ingestion | N/A | ~35 min (12K chunks, CPU) | Same |
| Documents supported | 500+ (GCS) | Limited by disk | Limited by disk |

---

## Credits

- **Organization**: Jabatan Kehakiman Syariah Terengganu (JKST)
- **Development**: Pendakwah Teknologi
- **AI Models**: Google Gemini, Mesolitica (Malaysian AI), Cohere, OpenAI
- **Embedding Model**: [Mesolitica Malaysian Embedding](https://huggingface.co/collections/mesolitica/malaysian-embedding-6523612bfe5881ad35f81b99)

## License

Internal use — Jabatan Kehakiman Syariah Terengganu & Pendakwah Teknologi.
