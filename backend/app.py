from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import asyncio
import hashlib
import time
import re
import csv
import glob
import fcntl
import base64
import io
import json
from datetime import datetime
import httpx

# RAG Mode Configuration
RAG_MODE = os.environ.get("RAG_MODE", "google")

# Import all SDKs — needed for per-request mode switching (compare page)
import urllib.parse
try:
    from google.cloud import storage
    from google import genai
    from google.oauth2 import service_account
    import google.auth.transport.requests
except ImportError:
    storage = None
    genai = None
    service_account = None
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None
try:
    import cohere
except ImportError:
    cohere = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Import providers module
from providers import (
    RAG_MODE as PROVIDER_MODE,
    get_providers, get_mode_info,
    ingest_knowledge_to_chroma
)

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "jab-kehakiman-syariah-tgg")
RAG_LOCATION = os.environ.get("GCP_RAG_LOCATION", "asia-southeast1")
GEMINI_LOCATION = os.environ.get("GCP_GEMINI_LOCATION", "us-central1")
RAG_CORPUS_ID = os.environ.get("GCP_RAG_CORPUS_ID", "6917529027641081856")
GEMINI_MODEL = os.environ.get("GCP_GEMINI_MODEL", "gemini-2.0-flash-001")

# Tavily API Configuration
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Cache Configuration
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "300"))  # 5 minutes default

# Reranking Configuration
ENABLE_RERANKING = os.environ.get("ENABLE_RERANKING", "true").lower() == "true"
RAG_INITIAL_RESULTS = int(os.environ.get("RAG_INITIAL_RESULTS", "15"))  # Retrieve more for reranking
RAG_FINAL_RESULTS = int(os.environ.get("RAG_FINAL_RESULTS", "5"))  # After reranking
RERANKER_TYPE = os.environ.get("RERANKER_TYPE", "gemini")  # 'gemini' or 'cohere'

# Cohere API Configuration
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

CREDENTIALS_PATH = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/opt/jkst-ai/jkst-credentials.json"
)

# GCS Bucket for documents
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "jkst-documents")

# Base URL for download links - set this to your public API URL
# Example: https://jkst.pendakwah.tech or http://your-server:8001
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8001")

# Local Knowledge Files Configuration
# Folder containing .txt files for additional knowledge
LOCAL_KNOWLEDGE_PATH = os.environ.get("LOCAL_KNOWLEDGE_PATH", "/opt/jkst-ai/knowledge")

# CSV Logging Configuration
# Path for conversation log CSV file
CSV_LOG_PATH = os.environ.get("CSV_LOG_PATH", "/opt/jkst-ai/logs/conversations.csv")

# Feedback Configuration
# Path for feedback CSV file
FEEDBACK_CSV_PATH = os.environ.get("FEEDBACK_CSV_PATH", "/opt/jkst-ai/logs/feedback.csv")

# ============================================================================
# TELEGRAM BOT CONFIGURATION
# ============================================================================
# Get your bot token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
# Webhook secret for verification (optional, but recommended)
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "jkst-webhook-secret-2024")
# Your public server URL (needed for webhook setup)
TELEGRAM_WEBHOOK_URL = os.environ.get("TELEGRAM_WEBHOOK_URL", "")  # e.g., https://your-domain.com

# ============================================================================
# VOICE FEATURES CONFIGURATION (STT + TTS)
# ============================================================================
# OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_openai_client = None

# Google Cloud TTS Configuration
TTS_LANGUAGE_CODE = os.environ.get("TTS_LANGUAGE_CODE", "ms-MY")
TTS_VOICE_NAME = os.environ.get("TTS_VOICE_NAME", "ms-MY-Wavenet-A")  # Female voice

# Initialize credentials
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
_credentials = None
_gemini_client = None
_tavily_client = None
_storage_client = None
_cohere_client = None

# ============================================================================
# FEATURE 1: IN-MEMORY CACHE
# ============================================================================
class SimpleCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._ttl = ttl_seconds

    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        hashed = self._hash_key(key)
        if hashed in self._cache:
            value, timestamp = self._cache[hashed]
            if time.time() - timestamp < self._ttl:
                return value
            else:
                del self._cache[hashed]
        return None

    def set(self, key: str, value: Any):
        hashed = self._hash_key(key)
        self._cache[hashed] = (value, time.time())

    def clear_expired(self):
        """Remove expired entries."""
        now = time.time()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts >= self._ttl]
        for k in expired:
            del self._cache[k]

    def stats(self) -> Dict[str, int]:
        self.clear_expired()
        return {"entries": len(self._cache), "ttl_seconds": self._ttl}


# Initialize caches
rag_cache = SimpleCache(ttl_seconds=CACHE_TTL_SECONDS)
response_cache = SimpleCache(ttl_seconds=CACHE_TTL_SECONDS)

# ============================================================================
# FEATURE 2: QUERY CLASSIFICATION
# ============================================================================
# Keywords that indicate internal JKST queries (no web search needed)
INTERNAL_KEYWORDS = [
    # JKST specific
    "jkst", "jkstr", "jabatan kehakiman syariah terengganu", "mahkamah syariah",
    # Shariah court matters
    "syariah", "syarie", "islam", "hukum syarak", "fatwa", "muamalat",
    # Court services
    "mahkamah", "bicara", "perbicaraan", "kes", "pendaftaran", "guaman",
    "tuntutan", "rayuan", "permohonan", "keputusan mahkamah",
    # Family law
    "perkahwinan", "kahwin", "nikah", "cerai", "perceraian", "talak",
    "fasakh", "khuluk", "nafkah", "hadhanah", "mut'ah", "harta sepencarian",
    # Mediation & counseling
    "sulh", "kaunseling", "rundingan", "pendamai", "mediasi",
    # Documents & procedures
    "dokumen", "borang", "sijil", "surat", "prosedur", "sop", "arahan",
    "garis panduan", "manual",
    # Departments
    "bahagian", "jabatan", "unit", "pengurusan", "pentadbiran",
    # HR & Admin
    "cuti", "tuntutan", "elaun", "gaji", "kakitangan", "pekerja",
    # Specific services
    "sokongan keluarga", "penyelidikan", "rekod", "teknologi maklumat",
]

# Keywords that indicate external/general queries (web search useful)
EXTERNAL_KEYWORDS = [
    "cuaca", "jadual", "waktu", "tarikh",
    "perbandingan", "statistik", "malaysia",
    "undang-undang sivil", "akta am", "peraturan kerajaan",
]

# Keywords that indicate JKST news/activities queries (fetch from official website)
JKST_NEWS_KEYWORDS = [
    "aktiviti terkini", "aktiviti jkst", "aktiviti terbaru",
    "berita terkini", "berita jkst", "berita terbaru", "berita semasa",
    "program terkini", "program jkst", "program terbaru",
    "majlis terkini", "majlis jkst",
    "acara terkini", "acara jkst",
    "kemas kini", "update terkini",
    "apa yang berlaku", "apa berlaku",
    "perkembangan terkini", "perkembangan jkst",
    "latest news", "latest activities",
]

# JKST Official News URLs
JKST_NEWS_URLS = [
    "https://syariah.terengganu.gov.my/index.php/arkib2",
    "https://syariah.terengganu.gov.my/index.php/arkib2/berita-semasa",
]


def classify_query(query: str) -> str:
    """
    Classify query as 'internal' (JKST-specific), 'external' (needs web search),
    'jkst_news' (fetch from official JKST website), or 'hybrid'.
    Returns: 'internal', 'external', 'jkst_news', or 'hybrid'
    """
    query_lower = query.lower()

    # Check for JKST news/activities queries FIRST
    jkst_news_score = sum(1 for kw in JKST_NEWS_KEYWORDS if kw in query_lower)
    if jkst_news_score >= 1:
        return "jkst_news"

    # Check for news patterns about JKST
    news_patterns = [
        r"(aktiviti|berita|program|majlis|acara).*(terkini|terbaru|semasa|jkst)",
        r"(terkini|terbaru|semasa).*(aktiviti|berita|program|majlis|acara)",
        r"apa.*(berlaku|terjadi|baru).*(jkst|mahkamah syariah)",
        r"(jkst|mahkamah syariah).*(buat|lakukan|anjur|adakan)",
    ]
    for pattern in news_patterns:
        if re.search(pattern, query_lower):
            return "jkst_news"

    internal_score = sum(1 for kw in INTERNAL_KEYWORDS if kw in query_lower)
    external_score = sum(1 for kw in EXTERNAL_KEYWORDS if kw in query_lower)

    # Strong internal indicators
    if internal_score >= 2 and external_score == 0:
        return "internal"

    # Strong external indicators
    if external_score >= 2 and internal_score == 0:
        return "external"

    # Check for question patterns that suggest internal knowledge
    internal_patterns = [
        r"apakah (prosedur|sop|polisi|arahan)",
        r"bagaimana (untuk|cara|proses)",
        r"siapa (bertanggungjawab|perlu)",
        r"bila (perlu|harus|mesti)",
        r"dokumen apa",
        r"borang (apa|mana)",
        r"bagaimana (mohon|daftar|memfailkan)",
    ]

    for pattern in internal_patterns:
        if re.search(pattern, query_lower):
            return "internal"

    # Default to hybrid (search both)
    return "hybrid"


# ============================================================================
# CREDENTIAL & CLIENT MANAGEMENT
# ============================================================================
def get_credentials():
    """Get and refresh Google Cloud credentials."""
    global _credentials
    if _credentials is None:
        _credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_PATH,
            scopes=SCOPES
        )
    if not _credentials.valid:
        auth_req = google.auth.transport.requests.Request()
        _credentials.refresh(auth_req)
    return _credentials


def get_gemini_client():
    """Get Gemini client (initialized once)."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=GEMINI_LOCATION
        )
    return _gemini_client


def get_tavily_client():
    """Get Tavily client for web search."""
    global _tavily_client
    if _tavily_client is None and TAVILY_API_KEY:
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


def get_cohere_client():
    """Get Cohere client for reranking."""
    global _cohere_client
    if _cohere_client is None and COHERE_API_KEY:
        _cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
    return _cohere_client


def get_storage_client():
    """Get Google Cloud Storage client."""
    global _storage_client
    if _storage_client is None:
        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_PATH
        )
        _storage_client = storage.Client(credentials=credentials, project=PROJECT_ID)
    return _storage_client


def get_openai_client():
    """Get OpenAI client (initialized once)."""
    global _openai_client
    if _openai_client is None and OPENAI_API_KEY:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def synthesize_speech_google(text: str) -> bytes:
    """
    Synthesize speech using Google Cloud Text-to-Speech.
    Returns audio bytes in MP3 format.
    """
    from google.cloud import texttospeech

    credentials = get_credentials()
    client = texttospeech.TextToSpeechClient(credentials=credentials)

    # Set the text input
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code=TTS_LANGUAGE_CODE,
        name=TTS_VOICE_NAME
    )

    # Select audio format
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,  # Normal speed
        pitch=0.0  # Normal pitch
    )

    # Perform the TTS request
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return response.audio_content


def gcs_uri_to_download_url(gcs_uri: str) -> str:
    """Convert gs:// URI to downloadable API URL with full base URL."""
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        return ""

    # Extract path from gs://bucket-name/path/to/file.pdf
    # More robust: extract everything after the bucket name (handles any bucket)
    try:
        # Remove gs:// prefix
        uri_without_scheme = gcs_uri[5:]  # Remove "gs://"
        # Find first slash after bucket name
        slash_index = uri_without_scheme.find('/')
        if slash_index > 0:
            path = uri_without_scheme[slash_index + 1:]
        else:
            path = uri_without_scheme
    except Exception:
        # Fallback to old method
        path = gcs_uri.replace(f"gs://{GCS_BUCKET_NAME}/", "")

    # URL encode the path
    encoded_path = urllib.parse.quote(path, safe="")

    # Return full download URL (so it opens in new tab, not routed by frontend)
    return f"{BASE_URL}/api/download/{encoded_path}"


# ============================================================================
# LOCAL KNOWLEDGE FILES (TXT AND MD FILES)
# ============================================================================
def read_local_knowledge_files() -> List[Dict[str, str]]:
    """
    Read all .txt and .md files from the local knowledge folder.
    Returns list of dicts with filename and content.
    """
    knowledge_files = []

    if not os.path.exists(LOCAL_KNOWLEDGE_PATH):
        print(f"Local knowledge folder not found: {LOCAL_KNOWLEDGE_PATH}")
        return knowledge_files

    try:
        # Find all .txt and .md files in the folder
        txt_files = glob.glob(os.path.join(LOCAL_KNOWLEDGE_PATH, "*.txt"))
        md_files = glob.glob(os.path.join(LOCAL_KNOWLEDGE_PATH, "*.md"))
        all_files = txt_files + md_files

        for knowledge_file in all_files:
            try:
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        filename = os.path.basename(knowledge_file)
                        knowledge_files.append({
                            "filename": filename,
                            "content": content
                        })
                        print(f"Loaded local knowledge: {filename} ({len(content)} chars)")
            except Exception as e:
                print(f"Error reading {knowledge_file}: {e}")

        print(f"Total local knowledge files loaded: {len(knowledge_files)}")

    except Exception as e:
        print(f"Error reading local knowledge folder: {e}")

    return knowledge_files


def search_local_knowledge(query: str) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Search local knowledge files for relevant content.
    Uses improved keyword matching to find relevant files.
    """
    import re
    knowledge_files = read_local_knowledge_files()

    if not knowledge_files:
        return [], []

    contexts = []
    sources = []

    # Clean and normalize query - remove punctuation and convert to lowercase
    query_cleaned = re.sub(r'[^\w\s]', '', query.lower())
    query_words = set(query_cleaned.split())

    # Define important keywords that should be weighted higher
    important_keywords = {
        'objektif', 'visi', 'misi', 'fungsi', 'struktur', 'organisasi',
        'pengenalan', 'sejarah', 'alamat', 'lokasi', 'waktu', 'operasi',
        'borang', 'permohonan', 'prosedur', 'sulh', 'cerai', 'nikah',
        'poligami', 'fasakh', 'nafkah', 'hadhanah', 'harta', 'pusaka',
        'waris', 'wasiat', 'hibah', 'wakaf', 'rayuan', 'mahkamah'
    }

    # Score each file based on keyword matches
    scored_files = []
    for kf in knowledge_files:
        content_lower = kf["content"].lower()
        filename_lower = kf["filename"].lower().replace('-', ' ').replace('_', ' ')

        # Count keyword matches
        score = 0
        matched_important = False

        for word in query_words:
            if len(word) > 2:  # Skip short words
                # Check if word is important keyword
                is_important = word in important_keywords

                # Content matches - count occurrences
                word_count = content_lower.count(word)
                if word_count > 0:
                    if is_important:
                        score += word_count * 20  # High weight for important keywords
                        matched_important = True
                    else:
                        score += word_count * 2  # Normal weight

                # Filename matches (weighted higher)
                if word in filename_lower:
                    if is_important:
                        score += 50  # Very high weight for important keywords in filename
                    else:
                        score += 10

        # If an important keyword was matched, boost the overall score
        if matched_important:
            score *= 2

        if score > 0:
            scored_files.append((score, kf))
            print(f"Local knowledge match: {kf['filename']} (score: {score})")

    # Sort by score and take top 5 (increased from 3)
    scored_files.sort(key=lambda x: x[0], reverse=True)
    top_files = scored_files[:5]

    for score, kf in top_files:
        # Truncate content for context
        content_preview = kf["content"][:1500] if len(kf["content"]) > 1500 else kf["content"]

        contexts.append(f"[SUMBER KEDUA - FAIL PENGETAHUAN TEMPATAN: {kf['filename']}]\n{content_preview}")
        sources.append({
            "type": "Fail Pengetahuan Tempatan",
            "filename": kf["filename"],
            "page_content": content_preview[:500],
            "score": score,
            "source_uri": f"local://{kf['filename']}",
            "priority": "SECONDARY"
        })

    return contexts, sources


# ============================================================================
# CSV CONVERSATION LOGGING
# ============================================================================
def ensure_csv_log_exists():
    """Ensure the CSV log file and directory exist with headers."""
    log_dir = os.path.dirname(CSV_LOG_PATH)

    # Create directory if it doesn't exist
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created log directory: {log_dir}")

    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'session_id',
                'question',
                'answer',
                'query_type',
                'rag_sources_count',
                'web_sources_count',
                'local_sources_count',
                'response_time_ms'
            ])
        print(f"Created conversation log: {CSV_LOG_PATH}")


def log_conversation(
    question: str,
    answer: str,
    query_type: str = "",
    rag_sources_count: int = 0,
    web_sources_count: int = 0,
    local_sources_count: int = 0,
    response_time_ms: int = 0,
    session_id: str = ""
):
    """
    Log a conversation to the CSV file.
    """
    try:
        ensure_csv_log_exists()

        # Generate session ID if not provided (based on timestamp)
        if not session_id:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Clean up text for CSV (remove newlines, limit length)
        clean_question = question.replace('\n', ' ').replace('\r', '')[:500]
        clean_answer = answer.replace('\n', ' ').replace('\r', '')[:2000]

        with open(CSV_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                session_id,
                clean_question,
                clean_answer,
                query_type,
                rag_sources_count,
                web_sources_count,
                local_sources_count,
                response_time_ms
            ])

        print(f"Logged conversation: {clean_question[:50]}...")

    except Exception as e:
        print(f"Error logging conversation: {e}")


app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    mode: Optional[str] = None  # Override RAG_MODE per-request: google, local, ultra


class ChatResponse(BaseModel):
    reply: str
    retrieval: Optional[List[Dict[str, Any]]] = None
    query_type: Optional[str] = None  # internal, external, or hybrid
    cache_hit: Optional[bool] = None
    mode: Optional[str] = None  # google, local, ultra
    self_evaluation: Optional[Dict[str, Any]] = None  # ultra mode only
    followup_suggestions: Optional[List[str]] = None  # ultra mode only


class FeedbackRequest(BaseModel):
    message_id: str
    timestamp: str
    user_question: str
    ai_response: str
    rating: Optional[str] = None  # 'thumbs_up' or 'thumbs_down'
    comment: Optional[str] = None
    retrieval_sources: int = 0
    retrieval_data: Optional[List[Dict[str, Any]]] = None


# ============================================================================
# RERANKING (COHERE OR GEMINI)
# ============================================================================
def rerank_with_cohere(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank retrieved documents using Cohere Rerank API.

    Cohere Rerank is a specialized cross-encoder model optimized for reranking.
    """
    if not documents or len(documents) <= RAG_FINAL_RESULTS:
        return documents

    try:
        client = get_cohere_client()
        if not client:
            print("Cohere client not available, falling back to original order")
            return documents[:RAG_FINAL_RESULTS]

        # Prepare documents for Cohere - combine filename and content
        docs_for_rerank = []
        for doc in documents:
            text = doc.get("page_content", "")[:1000]  # Cohere recommends <1000 chars
            filename = doc.get("filename", "")
            docs_for_rerank.append(f"{filename}\n{text}")

        # Call Cohere Rerank API
        response = client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=docs_for_rerank,
            top_n=RAG_FINAL_RESULTS
        )

        # Reorder documents based on Cohere ranking
        reranked = []
        for result in response.results:
            idx = result.index
            if 0 <= idx < len(documents):
                doc = documents[idx].copy()
                doc["rerank_score"] = result.relevance_score
                reranked.append(doc)

        print(f"Cohere reranking successful: {len(documents)} docs → top {len(reranked)}")
        for i, doc in enumerate(reranked[:3]):
            print(f"  #{i+1}: {doc.get('filename', 'Unknown')[:50]} (score: {doc.get('rerank_score', 0):.3f})")

        return reranked

    except Exception as e:
        print(f"Cohere reranking error: {e}")
        return documents[:RAG_FINAL_RESULTS]


def rerank_with_gemini(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank retrieved documents using Gemini for better relevance.

    This acts as a second-stage ranker that considers semantic relevance
    more deeply than the initial vector search.
    """
    if not documents or len(documents) <= RAG_FINAL_RESULTS:
        return documents

    try:
        client = get_gemini_client()

        # Build document list for ranking
        doc_list = []
        for i, doc in enumerate(documents):
            text_preview = doc.get("page_content", "")[:300]  # First 300 chars
            filename = doc.get("filename", "Unknown")
            doc_list.append(f"[DOC {i+1}] {filename}\n{text_preview}")

        docs_text = "\n\n---\n\n".join(doc_list)

        rerank_prompt = f"""Anda adalah sistem reranking dokumen. Tugas anda adalah menilai dan menyusun semula dokumen mengikut kerelevanan dengan soalan pengguna.

SOALAN PENGGUNA: {query}

DOKUMEN YANG DIPEROLEHI:
{docs_text}

ARAHAN:
1. Nilai setiap dokumen berdasarkan kerelevanan dengan soalan
2. Pertimbangkan:
   - Adakah dokumen menjawab soalan secara langsung?
   - Adakah dokumen mengandungi maklumat berkaitan prosedur/proses yang ditanya?
   - Adakah dokumen mengandungi borang/template yang diperlukan?
   - Adakah istilah/kata kunci dalam soalan wujud dalam dokumen?

3. Susun semula dokumen dari PALING RELEVAN ke KURANG RELEVAN

PENTING: Jawab HANYA dengan nombor dokumen dalam susunan kerelevanan, dipisahkan dengan koma.
Contoh jawapan: 3, 1, 5, 2, 4

SUSUNAN TERBAIK (nombor sahaja):"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=rerank_prompt,
            config={
                "temperature": 0.1,  # Low temperature for consistent ranking
                "max_output_tokens": 100
            }
        )

        # Parse the response to get ranking order
        ranking_text = response.text.strip()
        print(f"Gemini reranking response: {ranking_text}")

        # Extract numbers from response
        numbers = re.findall(r'\d+', ranking_text)

        if numbers:
            # Reorder documents based on ranking
            reranked = []
            seen = set()
            for num_str in numbers:
                idx = int(num_str) - 1  # Convert to 0-based index
                if 0 <= idx < len(documents) and idx not in seen:
                    reranked.append(documents[idx])
                    seen.add(idx)
                    if len(reranked) >= RAG_FINAL_RESULTS:
                        break

            # If we got valid reranking, use it
            if len(reranked) >= min(RAG_FINAL_RESULTS, len(documents)):
                print(f"Gemini reranking successful: {len(documents)} docs → top {len(reranked)}")
                return reranked

        # Fallback: return original order
        print("Gemini reranking fallback: using original order")
        return documents[:RAG_FINAL_RESULTS]

    except Exception as e:
        print(f"Gemini reranking error: {e}")
        # Fallback to original ranking
        return documents[:RAG_FINAL_RESULTS]


def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank documents using the configured reranker (Cohere or Gemini).
    """
    if RERANKER_TYPE == "cohere":
        return rerank_with_cohere(query, documents)
    else:
        return rerank_with_gemini(query, documents)


# ============================================================================
# RAG RETRIEVAL
# ============================================================================
async def retrieve_from_rag(query: str) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Retrieve relevant contexts from Google RAG corpus (PDFs) - PRIMARY SOURCE OF TRUTH.

    Flow:
    1. Retrieve initial set of documents (RAG_INITIAL_RESULTS, default 15)
    2. Rerank using Gemini for better relevance
    3. Return top documents (RAG_FINAL_RESULTS, default 5)
    """

    # Check cache first
    cached = rag_cache.get(f"rag:{query}")
    if cached:
        return cached

    credentials = get_credentials()

    rag_url = f"https://{RAG_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{RAG_LOCATION}:retrieveContexts"

    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
    }

    payload = {
        "vertexRagStore": {
            "ragCorpora": [f"projects/{PROJECT_ID}/locations/{RAG_LOCATION}/ragCorpora/{RAG_CORPUS_ID}"]
        },
        "query": {"text": query}
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(rag_url, headers=headers, json=payload)

    all_sources = []

    if response.status_code == 200:
        data = response.json()
        # Retrieve more documents for reranking (RAG_INITIAL_RESULTS)
        for ctx in data.get('contexts', {}).get('contexts', [])[:RAG_INITIAL_RESULTS]:
            text = ctx.get('text', '')
            if text:
                source_uri = ctx.get('sourceUri', '')
                filename = source_uri.split('/')[-1] if source_uri else 'Unknown'
                download_url = gcs_uri_to_download_url(source_uri)

                all_sources.append({
                    "type": "Dokumen Rasmi (RAG)",
                    "filename": filename,
                    "page_content": text,
                    "score": ctx.get('score', 0),
                    "source_uri": source_uri,
                    "download_url": download_url,
                    "priority": "PRIMARY"
                })

    # Apply reranking if enabled and we have documents
    if ENABLE_RERANKING and len(all_sources) > RAG_FINAL_RESULTS:
        print(f"Reranking {len(all_sources)} documents with {RERANKER_TYPE} for query: {query[:50]}...")
        reranked_sources = await asyncio.to_thread(rerank_documents, query, all_sources)
    else:
        reranked_sources = all_sources[:RAG_FINAL_RESULTS]

    # Build contexts from reranked sources
    contexts = []
    final_sources = []
    for src in reranked_sources:
        text = src.get("page_content", "")
        filename = src.get("filename", "Unknown")
        download_url = src.get("download_url", "")

        # Add download URL to context for AI to reference
        contexts.append(f"[SUMBER UTAMA - DOKUMEN RASMI: {filename}]\nMuat turun: {download_url}\n{text}")

        # Truncate page_content for response (keep full for reranking)
        final_sources.append({
            **src,
            "page_content": text[:500]
        })

    result = (contexts, final_sources)
    rag_cache.set(f"rag:{query}", result)
    return result


# ============================================================================
# WEB SEARCH (TAVILY)
# ============================================================================
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")


def search_web_brave(query: str) -> tuple[List[str], List[Dict[str, Any]]]:
    """Search the web using Brave Search API."""
    if not BRAVE_API_KEY:
        return [], []

    try:
        search_query = f"mahkamah syariah terengganu {query}" if "jkst" not in query.lower() and "syariah" not in query.lower() else query

        response = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": search_query, "count": 5},
            headers={"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"},
            timeout=15.0
        )
        response.raise_for_status()
        data = response.json()

        contexts = []
        sources = []

        for result in data.get("web", {}).get("results", [])[:5]:
            title = result.get("title", "")
            url = result.get("url", "")
            description = result.get("description", "")

            if title and description:
                contexts.append(f"[SUMBER TAMBAHAN - WEB] {title}\nURL: {url}\n{description}")
                sources.append({
                    "type": "Web (Brave)",
                    "filename": title,
                    "page_content": description[:500],
                    "score": 0,
                    "source_uri": url,
                    "priority": "SUPPLEMENTARY"
                })

        print(f"Brave search: {len(contexts)} results for: {search_query[:50]}")
        return contexts, sources

    except Exception as e:
        print(f"Brave search error: {e}")
        return [], []


def search_web_tavily(query: str) -> tuple[List[str], List[Dict[str, Any]]]:
    """Search the web using Tavily - SUPPLEMENTARY SOURCE ONLY."""
    try:
        client = get_tavily_client()
        if not client:
            return [], []

        # Search with Tavily - focus on JKST/Shariah court content when possible
        search_query = f"mahkamah syariah terengganu {query}" if "jkst" not in query.lower() and "syariah" not in query.lower() else query

        response = client.search(
            query=search_query,
            search_depth="advanced",
            max_results=5,
            include_answer=False,
            include_raw_content=False
        )

        contexts = []
        sources = []

        for result in response.get('results', [])[:5]:
            title = result.get('title', 'No Title')
            url = result.get('url', '')
            content = result.get('content', '')

            if title and content:
                contexts.append(f"[SUMBER TAMBAHAN - WEB] {title}\nURL: {url}\n{content}")
                sources.append({
                    "type": "Web (Tavily)",
                    "filename": title,
                    "page_content": content[:500],
                    "score": result.get('score', 0),
                    "source_uri": url,
                    "priority": "SUPPLEMENTARY"
                })

        return contexts, sources

    except Exception as e:
        print(f"Tavily search error: {e}")
        return [], []


def search_web(query: str) -> tuple[List[str], List[Dict[str, Any]]]:
    """Search the web using best available provider: Tavily → Brave → empty."""
    # Try Tavily first (better AI-structured results)
    if TAVILY_API_KEY:
        contexts, sources = search_web_tavily(query)
        if contexts:
            return contexts, sources

    # Fallback to Brave
    if BRAVE_API_KEY:
        contexts, sources = search_web_brave(query)
        if contexts:
            return contexts, sources

    return [], []


# ============================================================================
# JKST OFFICIAL NEWS SCRAPING
# ============================================================================
# Cache for JKST news (longer TTL since news doesn't change frequently)
jkst_news_cache = SimpleCache(ttl_seconds=1800)  # 30 minutes cache


async def fetch_jkst_news() -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Fetch latest news and activities from JKST official website.
    Returns contexts and sources for the AI to use.
    """
    # Check cache first
    cached = jkst_news_cache.get("jkst_news")
    if cached:
        print("JKST news cache hit")
        return cached

    contexts = []
    sources = []
    base_url = "https://syariah.terengganu.gov.my"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for url in JKST_NEWS_URLS:
                try:
                    print(f"Fetching JKST news from: {url}")
                    response = await client.get(url, headers={
                        "User-Agent": "Mozilla/5.0 (compatible; JKST-AI-Bot/1.0)"
                    })
                    response.raise_for_status()
                    html = response.text

                    # Parse news items using regex (lightweight, no BeautifulSoup needed)
                    # Pattern for arkib2 page (table format)
                    if "/arkib2" in url and "/berita-semasa" not in url:
                        # Extract news from table rows
                        # Pattern: <a href="/index.php/arkib2/NUMBER-SLUG">TITLE</a>
                        pattern = r'<a\s+href="(/index\.php/arkib2/\d+-[^"]+)"[^>]*>([^<]+)</a>'
                        matches = re.findall(pattern, html, re.IGNORECASE)

                        for link, title in matches[:10]:  # Get top 10 news
                            title = title.strip()
                            if title and len(title) > 5:
                                full_url = f"{base_url}{link}"
                                contexts.append(
                                    f"[BERITA RASMI JKST] {title}\n"
                                    f"Sumber: Laman Web Rasmi JKST\n"
                                    f"URL: {full_url}"
                                )
                                sources.append({
                                    "type": "Berita Rasmi JKST",
                                    "filename": title,
                                    "page_content": f"Berita/Aktiviti: {title}",
                                    "score": 1.0,
                                    "source_uri": full_url,
                                    "priority": "PRIMARY"
                                })

                    # Pattern for berita-semasa page (article format)
                    elif "/berita-semasa" in url:
                        # Extract news articles with more details
                        # Look for article titles and dates
                        title_pattern = r'<h\d[^>]*>\s*<a\s+href="(/index\.php/arkib2/berita-semasa/[^"]+)"[^>]*>([^<]+)</a>'
                        matches = re.findall(title_pattern, html, re.IGNORECASE)

                        # Also try alternative pattern
                        if not matches:
                            title_pattern = r'href="(/index\.php/arkib2/berita-semasa/\d+-[^"]+)"[^>]*>([^<]+)</a>'
                            matches = re.findall(title_pattern, html, re.IGNORECASE)

                        for link, title in matches[:10]:
                            title = title.strip()
                            if title and len(title) > 5 and not title.startswith('http'):
                                full_url = f"{base_url}{link}"
                                # Avoid duplicates
                                if not any(s.get("source_uri") == full_url for s in sources):
                                    contexts.append(
                                        f"[BERITA SEMASA JKST] {title}\n"
                                        f"Sumber: Laman Web Rasmi JKST - Berita Semasa\n"
                                        f"URL: {full_url}"
                                    )
                                    sources.append({
                                        "type": "Berita Semasa JKST",
                                        "filename": title,
                                        "page_content": f"Berita Semasa: {title}",
                                        "score": 1.0,
                                        "source_uri": full_url,
                                        "priority": "PRIMARY"
                                    })

                except Exception as e:
                    print(f"Error fetching {url}: {e}")
                    continue

        # If we got results, cache them
        if contexts:
            result = (contexts, sources)
            jkst_news_cache.set("jkst_news", result)
            print(f"Fetched {len(contexts)} news items from JKST website")
            return result

        return [], []

    except Exception as e:
        print(f"JKST news fetch error: {e}")
        return [], []


# JKST Website pages to search for organizational information
# URLs updated based on actual JKST website structure (verified 2024)
JKST_WEBSITE_PAGES = [
    # 0-3: Profil section
    ("https://syariah.terengganu.gov.my/index.php/profil/perutusan-kps", "Pengenalan & Perutusan"),
    ("https://syariah.terengganu.gov.my/index.php/profil/dasar", "Visi, Misi & Objektif"),
    ("https://syariah.terengganu.gov.my/index.php/profil/carta-organisasi-2023", "Carta Organisasi"),
    ("https://syariah.terengganu.gov.my/index.php/profil/struktur-organisasi", "Struktur Organisasi"),
    # 4-8: Bahagian section
    ("https://syariah.terengganu.gov.my/index.php/bahagian-suk/bahagian-khidmat-pengurusan-sumber-manusia", "Bahagian Khidmat Pengurusan"),
    ("https://syariah.terengganu.gov.my/index.php/bahagian-suk/bahagian-kehakiman", "Bahagian Kehakiman"),
    ("https://syariah.terengganu.gov.my/index.php/bahagian-suk/bahagian-sulh", "Bahagian Sulh"),
    ("https://syariah.terengganu.gov.my/index.php/bahagian-suk/seksyen-bahagian-sokongan-keluarga-sbsk", "Bahagian Sokongan Keluarga"),
    ("https://syariah.terengganu.gov.my/index.php/bahagian-suk/khidmat-nasihat-rundingcara", "Khidmat Nasihat & Rundingcara"),
    # 9-10: Perkhidmatan section
    ("https://syariah.terengganu.gov.my/index.php/hubungi-kami/ap-2", "Borang-Borang Perkhidmatan"),
    ("https://syariah.terengganu.gov.my/index.php/hubungi-kami/statistik-tahunan-kes-mal-jenayah", "Statistik Kes"),
    # 11-12: Hubungi Kami section
    ("https://syariah.terengganu.gov.my/index.php/muat-turun/alamat-mahkamah-rendah-syariah-daerah-daerah", "Alamat Mahkamah"),
    ("https://syariah.terengganu.gov.my/index.php/muat-turun/waktu-operasi-jkstr", "Waktu Operasi"),
    # 13-15: Soalan Lazim section
    ("https://syariah.terengganu.gov.my/index.php/soalan-lazim/umum", "Soalan Lazim Umum"),
    ("https://syariah.terengganu.gov.my/index.php/soalan-lazim/mahkamah-tinggi-syariah", "Soalan Lazim Mahkamah Tinggi"),
    ("https://syariah.terengganu.gov.my/index.php/soalan-lazim/mahkamah-rendah-syariah", "Soalan Lazim Mahkamah Rendah"),
]

# Cache for JKST website content
jkst_website_cache = SimpleCache(ttl_seconds=3600)  # 1 hour cache


async def fetch_jkst_website_content(query: str) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Fetch relevant content from JKST official website based on query.
    Searches multiple pages and extracts relevant information.
    """
    query_lower = query.lower()
    contexts = []
    sources = []
    base_url = "https://syariah.terengganu.gov.my"

    # Determine which pages to fetch based on query keywords
    pages_to_fetch = []

    # Keyword mapping to relevant pages (indices match JKST_WEBSITE_PAGES)
    keyword_page_mapping = {
        ("profil", "sejarah", "pengenalan", "latar belakang", "tentang", "perutusan"): [0],  # Pengenalan
        ("visi", "misi", "objektif", "matlamat", "tujuan", "dasar"): [1],  # Visi, Misi & Objektif
        ("carta", "organisasi", "struktur", "pegawai", "kakitangan"): [2, 3],  # Carta & Struktur Organisasi
        ("bahagian", "unit", "jabatan", "pengurusan"): [4, 5, 6, 7, 8],  # All Bahagian pages
        ("sulh", "mediasi", "pengantaraan"): [6],  # Bahagian Sulh
        ("sokongan keluarga", "bsk", "nafkah"): [7],  # Bahagian Sokongan Keluarga
        ("nasihat", "rundingcara", "kaunseling"): [8],  # Khidmat Nasihat
        ("perkhidmatan", "khidmat", "servis", "borang"): [9],  # Borang Perkhidmatan
        ("statistik", "kes", "data"): [10],  # Statistik Kes
        ("hubungi", "alamat", "telefon", "emel", "lokasi", "mahkamah"): [11, 12],  # Alamat & Waktu Operasi
        ("waktu", "operasi", "jam", "buka"): [12],  # Waktu Operasi
        ("soalan", "lazim", "faq"): [13, 14, 15],  # All Soalan Lazim pages
        ("piagam", "pelanggan", "komitmen"): [1],  # Redirect to Visi/Misi (piagam not available on site)
    }

    for keywords, page_indices in keyword_page_mapping.items():
        if any(kw in query_lower for kw in keywords):
            for page_index in page_indices:
                if page_index not in pages_to_fetch:
                    pages_to_fetch.append(page_index)

    # If no specific keywords found, fetch profil and visi/misi as defaults
    if not pages_to_fetch:
        pages_to_fetch = [0, 1]  # Pengenalan and Visi/Misi/Objektif

    # Limit to max 4 pages to avoid too many requests
    pages_to_fetch = pages_to_fetch[:4]

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for page_index in pages_to_fetch:
                url, page_name = JKST_WEBSITE_PAGES[page_index]

                # Check cache first
                cache_key = f"jkst_page:{page_index}"
                cached = jkst_website_cache.get(cache_key)
                if cached:
                    print(f"JKST website cache hit for: {page_name}")
                    contexts.append(cached["context"])
                    sources.append(cached["source"])
                    continue

                try:
                    print(f"Fetching JKST website: {page_name} ({url})")
                    response = await client.get(url, headers={
                        "User-Agent": "Mozilla/5.0 (compatible; JKST-AI-Bot/1.0)"
                    })
                    response.raise_for_status()
                    html = response.text

                    # Extract main content - remove scripts, styles, navigation
                    # Simple extraction: find content between common markers
                    content = html

                    # Remove script and style tags
                    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<nav[^>]*>.*?</nav>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<header[^>]*>.*?</header>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content = re.sub(r'<footer[^>]*>.*?</footer>', '', content, flags=re.DOTALL | re.IGNORECASE)

                    # Remove HTML tags but keep text
                    content = re.sub(r'<[^>]+>', ' ', content)

                    # Clean up whitespace
                    content = re.sub(r'\s+', ' ', content).strip()

                    # Decode HTML entities
                    content = content.replace('&nbsp;', ' ')
                    content = content.replace('&amp;', '&')
                    content = content.replace('&lt;', '<')
                    content = content.replace('&gt;', '>')
                    content = content.replace('&quot;', '"')

                    # Limit content length
                    if len(content) > 2000:
                        content = content[:2000] + "..."

                    if content and len(content) > 100:
                        context_text = f"[LAMAN WEB RASMI JKST - {page_name}]\nURL: {url}\n\n{content}"
                        source_data = {
                            "type": "Laman Web Rasmi JKST",
                            "filename": page_name,
                            "page_content": content[:500],
                            "score": 1.0,
                            "source_uri": url,
                            "priority": "PRIMARY"
                        }

                        contexts.append(context_text)
                        sources.append(source_data)

                        # Cache the result
                        jkst_website_cache.set(cache_key, {
                            "context": context_text,
                            "source": source_data
                        })

                except Exception as e:
                    print(f"Error fetching {url}: {e}")
                    continue

        if contexts:
            print(f"Fetched {len(contexts)} pages from JKST official website")

        return contexts, sources

    except Exception as e:
        print(f"JKST website fetch error: {e}")
        return [], []


def format_jkst_news_for_prompt(contexts: List[str]) -> str:
    """Format JKST news contexts into a structured prompt section."""
    if not contexts:
        return ""

    news_text = "\n\n".join(contexts)
    return f"""
=== BERITA DAN AKTIVITI TERKINI JKST ===
(Sumber: Laman Web Rasmi JKST - https://syariah.terengganu.gov.my)

{news_text}

=== AKHIR BERITA JKST ===
"""


# ============================================================================
# PROMPT BUILDING
# ============================================================================
def build_prompt(query: str, rag_contexts: List[str], web_contexts: List[str], history: List[Dict[str, str]], query_type: str = "hybrid", local_contexts: List[str] = None, jkst_news_contexts: List[str] = None) -> str:
    """Build the prompt for Gemini."""

    # RAG contexts are PRIMARY - put them first and emphasize their importance
    rag_text = "\n\n---\n\n".join(rag_contexts) if rag_contexts else ""
    web_text = "\n\n---\n\n".join(web_contexts) if web_contexts else ""
    local_text = "\n\n---\n\n".join(local_contexts) if local_contexts else ""
    jkst_news_text = format_jkst_news_for_prompt(jkst_news_contexts) if jkst_news_contexts else ""

    # Build conversation history
    history_text = ""
    if len(history) > 1:
        for msg in history[:-1]:
            role = "Pengguna" if msg["role"] == "user" else "Pembantu"
            history_text += f"{role}: {msg['content']}\n"

    # Base system prompt — natural conversational style
    system_prompt = """Kamu adalah pegawai khidmat pelanggan JKST (Jabatan Kehakiman Syariah Terengganu) yang mesra dan berpengetahuan. Jawab seperti manusia biasa yang sedang berbual — bukan robot. Gunakan bahasa yang santai tapi profesional.

PERATURAN PENTING:
1. JANGAN SEKALI-KALI guna emoji, emotikon, atau simbol hiasan dalam jawapan.
2. Tulis secara semula jadi seperti manusia berbual di kaunter — ringkas, jelas, mesra.
3. Jangan guna ayat pembuka yang klise seperti "Terima kasih atas soalan anda" atau "Soalan yang bagus". Terus jawab.
4. Jangan ulang soalan pengguna balik. Terus beri jawapan.
5. Guna "kamu/anda" bukan "tuan/puan" kecuali konteks rasmi.

RUJUKAN DAN SUMBER:
- Jika jawapan berdasarkan dokumen tertentu, WAJIB sertakan rujukan yang boleh diklik.
- Format rujukan: [Nama Dokumen](URL_PENUH) — guna URL tepat dari konteks yang diberikan.
- Jika ada pautan muat turun, sertakan terus: Muat turun: [nama fail](URL)
- Jika tiada dokumen spesifik, nyatakan sumber secara umum (cth: "Berdasarkan maklumat dari laman web rasmi JKST").
- JANGAN reka URL. Guna HANYA URL yang ada dalam konteks.

BATASAN:
- Jangan beri nasihat undang-undang, fatwa, atau tafsiran hukum syarak.
- Untuk perkara undang-undang, cadangkan jumpa pegawai mahkamah atau peguam syarie.
- Jika tak pasti atau maklumat tak dijumpai, cakap terus terang — jangan reka jawapan.
- Utamakan dokumen rasmi JKST. Maklumat web hanya sebagai tambahan.

MAKLUMAT HUBUNGAN:
Alamat: Tingkat 5, Bangunan Mahkamah Syariah, Jalan Sultan Mohamad, 21100 Kuala Terengganu
Waktu: Ahad-Rabu 8am-4pm, Khamis 8am-3pm
Tel: 09-623 2323 | Faks: 09-624 1510
Emel: jkstr@esyariah.gov.my
Web: https://syariah.terengganu.gov.my"""

    # Construct context section
    context_section = ""

    # JKST News comes first if available (from official website)
    if jkst_news_text:
        context_section += jkst_news_text

    if rag_text:
        context_section += f"""
=== DOKUMEN RASMI JKST (SUMBER UTAMA - WAJIB DIUTAMAKAN) ===
{rag_text}
"""

    if local_text:
        context_section += f"""
=== FAIL PENGETAHUAN TEMPATAN (SUMBER KEDUA) ===
{local_text}
"""

    if web_text:
        context_section += f"""
=== MAKLUMAT WEB (SUMBER TAMBAHAN SAHAJA) ===
{web_text}
"""

    if not context_section:
        context_section = "Tiada dokumen atau maklumat web berkaitan dijumpai."

    # Build final prompt
    if history_text:
        prompt = f"""{system_prompt}

SEJARAH PERBUALAN:
{history_text}

{context_section}

SOALAN SEMASA: {query}

JAWAPAN (utamakan dokumen rasmi, nyatakan sumber):"""
    else:
        prompt = f"""{system_prompt}

{context_section}

SOALAN: {query}

JAWAPAN (utamakan dokumen rasmi, nyatakan sumber):"""

    return prompt


# ============================================================================
# GEMINI GENERATION (NON-STREAMING)
# ============================================================================
def generate_with_gemini(query: str, rag_contexts: List[str], web_contexts: List[str], history: List[Dict[str, str]], query_type: str = "hybrid", local_contexts: List[str] = None, jkst_news_contexts: List[str] = None) -> str:
    """Generate response using Gemini with RAG as primary source."""
    client = get_gemini_client()
    prompt = build_prompt(query, rag_contexts, web_contexts, history, query_type, local_contexts, jkst_news_contexts)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "max_output_tokens": 2000
        }
    )

    return response.text


# ============================================================================
# FEATURE 3: STREAMING GENERATION
# ============================================================================
def generate_with_gemini_stream_sync(query: str, rag_contexts: List[str], web_contexts: List[str], history: List[Dict[str, str]], query_type: str = "hybrid", local_contexts: List[str] = None, jkst_news_contexts: List[str] = None):
    """Generate streaming response using Gemini (synchronous generator)."""
    client = get_gemini_client()
    prompt = build_prompt(query, rag_contexts, web_contexts, history, query_type, local_contexts, jkst_news_contexts)

    # Use streaming
    response_stream = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "max_output_tokens": 2000
        }
    )

    for chunk in response_stream:
        if chunk.text:
            yield chunk.text


# ============================================================================
# LOCAL/ULTRA MODE CHAT HANDLERS
# ============================================================================
async def chat_local_or_ultra(messages_payload: List[Dict[str, str]], latest_message: str, override_mode: str = None) -> Dict[str, Any]:
    """
    Handle chat for local and ultra modes using providers.
    Returns dict with reply, retrieval, query_type, and extras.
    """
    mode = override_mode or RAG_MODE
    # Temporarily set providers module mode for this request
    import providers as _prov
    old_mode = _prov.RAG_MODE
    _prov.RAG_MODE = mode
    try:
        providers = get_providers()
    finally:
        _prov.RAG_MODE = old_mode
    start_time = time.time()

    # Step 1: Classify query (same for all modes)
    query_type = classify_query(latest_message)
    print(f"[{mode.upper()}] Query classified as: {query_type}")

    # Step 2: Retrieve from local vector DB
    retriever = providers["retriever"]
    contexts, sources = await retriever.retrieve(latest_message)

    # Step 3: Also search local knowledge files (keyword-based, supplements vector search)
    local_contexts, local_sources = await asyncio.to_thread(
        search_local_knowledge, latest_message
    )

    # Step 4: Web search if needed (Tavily still available for all modes)
    web_contexts, web_sources = [], []
    jkst_news_contexts, jkst_news_sources = [], []

    if query_type == "jkst_news":
        jkst_news_contexts, jkst_news_sources = await fetch_jkst_news()
    elif query_type in ("external", "hybrid"):
        web_contexts, web_sources = await asyncio.to_thread(
            search_web, latest_message
        ) if TavilyClient else ([], [])

    # Step 5: Rerank if we have a reranker
    reranker = providers["reranker"]
    if reranker and len(sources) > 5:
        sources = await asyncio.to_thread(reranker.rerank, latest_message, sources, 5)
        contexts = [
            f"[SUMBER - DOKUMEN: {s.get('filename', '?')}]\n{s.get('page_content', '')}"
            for s in sources
        ]

    # Combine all sources
    all_sources = sources + local_sources + jkst_news_sources + web_sources

    # Step 6: Generate response
    generator = providers["generator"]
    enhancer = providers.get("enhancer")

    # Build prompt based on mode
    if mode == "ultra" and enhancer:
        # Ultra mode: use enhanced prompt with CoT
        expanded_queries = None
        if hasattr(retriever, 'expand_query'):
            # Already expanded during retrieval, just note it
            expanded_queries = ["(expanded during retrieval)"]

        prompt = enhancer.build_ultra_prompt(
            latest_message, contexts, web_contexts,
            messages_payload, query_type, local_contexts,
            jkst_news_contexts, expanded_queries
        )
    else:
        # Local mode: use standard prompt builder
        prompt = build_prompt(
            latest_message, contexts, web_contexts,
            messages_payload, query_type, local_contexts,
            jkst_news_contexts
        )

    reply = await asyncio.to_thread(generator.generate, prompt)

    # Step 7: Ultra mode extras — run self-eval + follow-ups IN PARALLEL
    extras = {}
    if mode == "ultra" and enhancer:
        # Run both LLM calls concurrently (saves ~8-10s)
        eval_task = asyncio.to_thread(
            enhancer.self_evaluate, latest_message, reply, contexts
        )
        followup_task = asyncio.to_thread(
            enhancer.suggest_followups, latest_message, reply
        )
        eval_data, followups = await asyncio.gather(eval_task, followup_task)

        extras["self_evaluation"] = eval_data
        extras["followup_suggestions"] = followups

        # Save to conversation memory
        memory = providers.get("memory")
        if memory:
            session_id = hashlib.md5(
                json.dumps(messages_payload[:1]).encode()
            ).hexdigest()[:12]
            memory.save_turn(session_id, "user", latest_message)
            memory.save_turn(session_id, "assistant", reply[:2000])

    # Log conversation
    response_time_ms = int((time.time() - start_time) * 1000)
    await asyncio.to_thread(
        log_conversation, latest_message, reply, query_type,
        len(sources), len(web_sources), len(local_sources), response_time_ms
    )

    return {
        "reply": reply,
        "retrieval": all_sources if all_sources else None,
        "query_type": query_type,
        "mode": mode,
        **extras
    }


async def stream_local_or_ultra(messages_payload: List[Dict[str, str]], latest_message: str, override_mode: str = None):
    """Streaming version for local/ultra modes."""
    mode = override_mode or RAG_MODE
    import providers as _prov
    old_mode = _prov.RAG_MODE
    _prov.RAG_MODE = mode
    try:
        providers = get_providers()
    finally:
        _prov.RAG_MODE = old_mode

    query_type = classify_query(latest_message)

    # Retrieve
    retriever = providers["retriever"]
    contexts, sources = await retriever.retrieve(latest_message)

    local_contexts, local_sources = await asyncio.to_thread(
        search_local_knowledge, latest_message
    )

    web_contexts, web_sources = [], []
    jkst_news_contexts = []

    if query_type == "jkst_news":
        jkst_news_contexts, _ = await fetch_jkst_news()
    elif query_type in ("external", "hybrid"):
        web_contexts, _ = await asyncio.to_thread(
            search_web, latest_message
        ) if TavilyClient else ([], [])

    # Rerank
    reranker = providers["reranker"]
    if reranker and len(sources) > 5:
        sources = await asyncio.to_thread(reranker.rerank, latest_message, sources, 5)
        contexts = [
            f"[SUMBER - DOKUMEN: {s.get('filename', '?')}]\n{s.get('page_content', '')}"
            for s in sources
        ]

    all_sources = sources + local_sources + web_sources
    enhancer = providers.get("enhancer")
    generator = providers["generator"]

    if mode == "ultra" and enhancer:
        prompt = enhancer.build_ultra_prompt(
            latest_message, contexts, web_contexts,
            messages_payload, query_type, local_contexts, jkst_news_contexts
        )
    else:
        prompt = build_prompt(
            latest_message, contexts, web_contexts,
            messages_payload, query_type, local_contexts, jkst_news_contexts
        )

    # Return sources + streaming generator
    return all_sources, query_type, generator.generate_stream(prompt)


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Standard chat endpoint with caching and query classification."""
    start_time = time.time()

    # Per-request mode override (for multi-interface demo)
    request_mode = req.mode if req.mode in ("google", "local", "ultra") else RAG_MODE

    messages_payload = [
        {"role": msg.role, "content": msg.content}
        for msg in req.messages
    ]

    latest_message = messages_payload[-1]["content"] if messages_payload else ""

    if not latest_message:
        raise HTTPException(status_code=400, detail="Mesej kosong")

    # ---- LOCAL / ULTRA MODE: use providers ----
    if request_mode in ("local", "ultra"):
        try:
            # Check cache first
            cache_key = f"response:{request_mode}:{latest_message}"
            cached_response = response_cache.get(cache_key)
            if cached_response:
                return ChatResponse(
                    reply=cached_response["reply"],
                    retrieval=cached_response["retrieval"],
                    query_type=cached_response.get("query_type"),
                    cache_hit=True,
                    mode=request_mode
                )

            result = await chat_local_or_ultra(messages_payload, latest_message, override_mode=request_mode)

            # Cache the response
            response_cache.set(cache_key, {
                "reply": result["reply"],
                "retrieval": result["retrieval"],
                "query_type": result.get("query_type")
            })

            return ChatResponse(
                reply=result["reply"],
                retrieval=result["retrieval"],
                query_type=result.get("query_type"),
                cache_hit=False,
                mode=request_mode,
                self_evaluation=result.get("self_evaluation"),
                followup_suggestions=result.get("followup_suggestions")
            )
        except Exception as exc:
            print(f"ERROR [{request_mode}]: {exc}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=502, detail=f"Ralat sistem ({RAG_MODE}): {str(exc)}")

    # ---- GOOGLE MODE: original logic ----
    # Check response cache
    cache_key = f"response:{latest_message}"
    cached_response = response_cache.get(cache_key)
    if cached_response:
        return ChatResponse(
            reply=cached_response["reply"],
            retrieval=cached_response["retrieval"],
            query_type=cached_response.get("query_type"),
            cache_hit=True
        )

    try:
        # Classify query
        query_type = classify_query(latest_message)
        print(f"Query classified as: {query_type}")

        # Step 1: Search RAG (PRIMARY source)
        rag_contexts, rag_sources = await retrieve_from_rag(latest_message)

        # Step 2: Search local knowledge files (SECONDARY source)
        local_contexts, local_sources = await asyncio.to_thread(
            search_local_knowledge, latest_message
        )

        # Step 3: Conditionally search web based on classification
        # Fallback chain: RAG → Local → JKST Website → General Web (Tavily)
        web_contexts, web_sources = [], []
        jkst_news_contexts, jkst_news_sources = [], []
        jkst_website_contexts, jkst_website_sources = [], []

        # Check if we have sufficient internal data (RAG + local)
        # Also check quality - if best rerank score is low, data may not be relevant
        has_sufficient_internal_data = len(rag_contexts) >= 2 or len(local_contexts) >= 2

        # Check if best RAG result has good relevance score (rerank_score > 0.15)
        best_rag_score = 0
        if rag_sources:
            best_rag_score = max(s.get('rerank_score', 0) for s in rag_sources)
        has_relevant_rag_data = best_rag_score >= 0.15

        # Keywords that indicate organizational info queries (should always check JKST website)
        org_info_keywords = [
            "piagam", "visi", "misi", "objektif", "carta", "organisasi",
            "struktur", "fungsi", "peranan", "sejarah", "latar belakang",
            "alamat", "lokasi", "hubungi", "telefon", "waktu operasi"
        ]
        is_org_info_query = any(kw in latest_message.lower() for kw in org_info_keywords)

        if query_type == "jkst_news":
            # Fetch latest news/activities from JKST official website
            print("Fetching JKST news from official website...")
            jkst_news_contexts, jkst_news_sources = await fetch_jkst_news()

        elif query_type == "internal":
            # For internal/JKST-related queries
            # Fetch from JKST website if: insufficient data, OR low relevance, OR org info query
            should_fetch_website = (not has_sufficient_internal_data) or (not has_relevant_rag_data) or is_org_info_query

            if should_fetch_website:
                # Step 3a: First try JKST official website
                print(f"Fetching from JKST website (sufficient_data={has_sufficient_internal_data}, relevant_rag={has_relevant_rag_data}, org_query={is_org_info_query})...")
                jkst_website_contexts, jkst_website_sources = await fetch_jkst_website_content(latest_message)

                # Step 3b: If still not enough AND no relevant RAG, try general web search
                if not jkst_website_contexts and not has_relevant_rag_data:
                    print("No JKST website results and low RAG relevance, falling back to Tavily web search...")
                    web_contexts, web_sources = await asyncio.to_thread(
                        search_web, latest_message
                    )

        elif query_type == "hybrid":
            # For hybrid queries, check internal data first
            should_fetch_website = (not has_sufficient_internal_data) or (not has_relevant_rag_data) or is_org_info_query

            if should_fetch_website:
                # Step 3a: First try JKST official website
                print(f"Hybrid query, fetching from JKST website (sufficient_data={has_sufficient_internal_data}, relevant_rag={has_relevant_rag_data}, org_query={is_org_info_query})...")
                jkst_website_contexts, jkst_website_sources = await fetch_jkst_website_content(latest_message)

            # Step 3b: Also search general web for hybrid queries
            print("Searching general web for hybrid query...")
            web_contexts, web_sources = await asyncio.to_thread(
                search_web, latest_message
            )

        elif query_type == "external":
            # For external queries, go straight to web search
            web_contexts, web_sources = await asyncio.to_thread(
                search_web, latest_message
            )

        # Combine all JKST website contexts
        all_jkst_contexts = jkst_news_contexts + jkst_website_contexts
        all_jkst_sources = jkst_news_sources + jkst_website_sources

        # Combine sources - JKST (news + website) first, RAG (primary), local (secondary), then web (supplementary)
        all_sources = all_jkst_sources + rag_sources + local_sources + web_sources

        # Step 4: Generate response with Gemini (pass query_type for special handling)
        reply = await asyncio.to_thread(
            generate_with_gemini,
            latest_message,
            rag_contexts,
            web_contexts,
            messages_payload,
            query_type,
            local_contexts,
            all_jkst_contexts
        )

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Log conversation to CSV
        await asyncio.to_thread(
            log_conversation,
            latest_message,
            reply,
            query_type,
            len(rag_sources),
            len(web_sources),
            len(local_sources),
            response_time_ms
        )

        # Cache the response
        response_cache.set(cache_key, {
            "reply": reply,
            "retrieval": all_sources if all_sources else None,
            "query_type": query_type
        })

        return ChatResponse(
            reply=reply,
            retrieval=all_sources if all_sources else None,
            query_type=query_type,
            cache_hit=False,
            mode="google"
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Permintaan tamat masa. Sila cuba semula."
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail=f"Ralat sistem: {str(exc)}"
        )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat endpoint for real-time responses."""
    start_time = time.time()

    request_mode = req.mode if req.mode in ("google", "local", "ultra") else RAG_MODE

    messages_payload = [
        {"role": msg.role, "content": msg.content}
        for msg in req.messages
    ]

    latest_message = messages_payload[-1]["content"] if messages_payload else ""

    if not latest_message:
        raise HTTPException(status_code=400, detail="Mesej kosong")

    # ---- LOCAL / ULTRA MODE: use providers for streaming ----
    if request_mode in ("local", "ultra"):
        try:
            all_sources, query_type, gen_stream = await stream_local_or_ultra(
                messages_payload, latest_message, override_mode=request_mode
            )

            async def local_stream_generator():
                # Send sources first
                import json as _json
                sources_event = _json.dumps({
                    "type": "sources",
                    "data": all_sources,
                    "query_type": query_type,
                    "mode": RAG_MODE
                })
                yield f"data: {sources_event}\n\n"

                # Stream LLM response
                full_response = []
                for text_chunk in gen_stream:
                    full_response.append(text_chunk)
                    chunk_data = _json.dumps({"type": "content", "data": text_chunk})
                    yield f"data: {chunk_data}\n\n"

                yield "data: [DONE]\n\n"

            return StreamingResponse(
                local_stream_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        except Exception as exc:
            print(f"ERROR [{RAG_MODE}] stream: {exc}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=502, detail=f"Ralat sistem ({RAG_MODE}): {str(exc)}")

    # ---- GOOGLE MODE: original streaming logic ----
    try:
        # Classify query
        query_type = classify_query(latest_message)

        # Step 1: Search RAG (PRIMARY source)
        rag_contexts, rag_sources = await retrieve_from_rag(latest_message)

        # Step 2: Search local knowledge files (SECONDARY source)
        local_contexts, local_sources = await asyncio.to_thread(
            search_local_knowledge, latest_message
        )

        # Step 3: Conditionally search web based on classification
        # Fallback chain: RAG → Local → JKST Website → General Web (Tavily)
        web_contexts, web_sources = [], []
        jkst_news_contexts, jkst_news_sources = [], []
        jkst_website_contexts, jkst_website_sources = [], []

        # Check if we have sufficient internal data (RAG + local)
        has_sufficient_internal_data = len(rag_contexts) >= 2 or len(local_contexts) >= 2

        # Check if best RAG result has good relevance score (rerank_score > 0.15)
        best_rag_score = 0
        if rag_sources:
            best_rag_score = max(s.get('rerank_score', 0) for s in rag_sources)
        has_relevant_rag_data = best_rag_score >= 0.15

        # Keywords that indicate organizational info queries (should always check JKST website)
        org_info_keywords = [
            "piagam", "visi", "misi", "objektif", "carta", "organisasi",
            "struktur", "fungsi", "peranan", "sejarah", "latar belakang",
            "alamat", "lokasi", "hubungi", "telefon", "waktu operasi"
        ]
        is_org_info_query = any(kw in latest_message.lower() for kw in org_info_keywords)

        if query_type == "jkst_news":
            # Fetch latest news/activities from JKST official website
            jkst_news_contexts, jkst_news_sources = await fetch_jkst_news()

        elif query_type == "internal":
            # For internal/JKST-related queries
            should_fetch_website = (not has_sufficient_internal_data) or (not has_relevant_rag_data) or is_org_info_query

            if should_fetch_website:
                # Step 3a: First try JKST official website
                jkst_website_contexts, jkst_website_sources = await fetch_jkst_website_content(latest_message)

                # Step 3b: If still not enough AND no relevant RAG, try general web search
                if not jkst_website_contexts and not has_relevant_rag_data:
                    web_contexts, web_sources = await asyncio.to_thread(
                        search_web, latest_message
                    )

        elif query_type == "hybrid":
            # For hybrid queries, check internal data first
            should_fetch_website = (not has_sufficient_internal_data) or (not has_relevant_rag_data) or is_org_info_query

            if should_fetch_website:
                # Step 3a: First try JKST official website
                jkst_website_contexts, jkst_website_sources = await fetch_jkst_website_content(latest_message)

            # Step 3b: Also search general web for hybrid queries
            web_contexts, web_sources = await asyncio.to_thread(
                search_web, latest_message
            )

        elif query_type == "external":
            # For external queries, go straight to web search
            web_contexts, web_sources = await asyncio.to_thread(
                search_web, latest_message
            )

        # Combine all JKST website contexts
        all_jkst_contexts = jkst_news_contexts + jkst_website_contexts
        all_jkst_sources = jkst_news_sources + jkst_website_sources

        # Combine all sources
        all_sources = all_jkst_sources + rag_sources + local_sources + web_sources

        # Stream the response
        import json

        def stream_generator():
            # First, send metadata as JSON
            metadata = {
                "type": "metadata",
                "query_type": query_type,
                "sources": all_sources
            }
            yield f"data: {json.dumps(metadata)}\n\n"

            # Collect full response for logging
            full_response = []

            # Then stream the text
            for chunk in generate_with_gemini_stream_sync(
                latest_message,
                rag_contexts,
                web_contexts,
                messages_payload,
                query_type,
                local_contexts,
                all_jkst_contexts
            ):
                full_response.append(chunk)
                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"

            # Log conversation after streaming completes
            response_time_ms = int((time.time() - start_time) * 1000)
            log_conversation(
                latest_message,
                "".join(full_response),
                query_type,
                len(rag_sources),
                len(web_sources),
                len(local_sources),
                response_time_ms
            )

            # Signal end
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as exc:
        print(f"ERROR: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail=f"Ralat sistem: {str(exc)}"
        )


@app.get("/api/health")
async def health():
    # Count local knowledge files (both .txt and .md)
    local_files_count = 0
    if os.path.exists(LOCAL_KNOWLEDGE_PATH):
        txt_count = len(glob.glob(os.path.join(LOCAL_KNOWLEDGE_PATH, "*.txt")))
        md_count = len(glob.glob(os.path.join(LOCAL_KNOWLEDGE_PATH, "*.md")))
        local_files_count = txt_count + md_count

    mode_info = get_mode_info()
    return {
        "status": "ok",
        "rag_mode": RAG_MODE,
        "mode_description": mode_info["description"],
        "backend": f"Google RAG (Primary) + Local Knowledge + Tavily Web Search + {GEMINI_MODEL}" if RAG_MODE == "google" else mode_info["description"],
        "rag_priority": "PRIMARY SOURCE OF TRUTH",
        "local_priority": "SECONDARY SOURCE",
        "web_priority": "SUPPLEMENTARY ONLY",
        "features": {
            "query_classification": True,
            "caching": True,
            "streaming": True,
            "reranking": ENABLE_RERANKING,
            "local_knowledge": True,
            "csv_logging": True
        },
        "local_knowledge_config": {
            "path": LOCAL_KNOWLEDGE_PATH,
            "files_count": local_files_count
        },
        "csv_logging_config": {
            "path": CSV_LOG_PATH
        },
        "reranking_config": {
            "enabled": ENABLE_RERANKING,
            "type": RERANKER_TYPE,
            "initial_results": RAG_INITIAL_RESULTS,
            "final_results": RAG_FINAL_RESULTS,
            "model": "rerank-v3.5" if RERANKER_TYPE == "cohere" else GEMINI_MODEL
        },
        "feedback_config": {
            "path": FEEDBACK_CSV_PATH,
            "enabled": True
        }
    }


@app.get("/api/mode")
async def get_current_mode():
    """Get current RAG mode and its features."""
    return get_mode_info()


@app.get("/api/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    return {
        "rag_cache": rag_cache.stats(),
        "response_cache": response_cache.stats()
    }


@app.post("/api/cache/clear")
async def cache_clear():
    """Clear all caches."""
    global rag_cache, response_cache
    rag_cache = SimpleCache(ttl_seconds=CACHE_TTL_SECONDS)
    response_cache = SimpleCache(ttl_seconds=CACHE_TTL_SECONDS)
    return {"status": "cleared"}


# ============================================================================
# DOCUMENT DOWNLOAD ENDPOINT
# ============================================================================
@app.get("/api/download/{file_path:path}")
async def download_document(file_path: str):
    """
    Download a document from GCS.

    Usage: GET /api/download/JKST/BORANG/BORANG%20PERMOHONAN%20CERAI.pdf
    """
    try:
        # Decode the URL-encoded path
        decoded_path = urllib.parse.unquote(file_path)

        # Strip gs://bucket-name/ prefix if present (backward compatibility)
        if decoded_path.startswith("gs://"):
            # Remove gs:// prefix and extract path after bucket name
            uri_without_scheme = decoded_path[5:]  # Remove "gs://"
            slash_index = uri_without_scheme.find('/')
            if slash_index > 0:
                decoded_path = uri_without_scheme[slash_index + 1:]

        # Get storage client and bucket
        client = get_storage_client()
        bucket = client.bucket(GCS_BUCKET_NAME)

        # Try original path first (may contain NBSP characters that match GCS filenames)
        blob = bucket.blob(decoded_path)

        # If not found, try with NBSP normalized to regular spaces
        if not blob.exists():
            import re
            normalized_path = decoded_path.replace('\u00a0', ' ')
            normalized_path = re.sub(r' +', ' ', normalized_path)
            if normalized_path != decoded_path:
                blob = bucket.blob(normalized_path)
                decoded_path = normalized_path  # Use normalized for filename extraction

        # Check if file exists
        if not blob.exists():
            # Log for debugging
            print(f"File not found in GCS bucket '{GCS_BUCKET_NAME}': '{decoded_path}'")
            raise HTTPException(status_code=404, detail=f"Dokumen tidak dijumpai: {decoded_path}")

        # Get file content
        content = blob.download_as_bytes()

        # Determine content type
        content_type = blob.content_type or "application/octet-stream"
        if decoded_path.lower().endswith(".pdf"):
            content_type = "application/pdf"
        elif decoded_path.lower().endswith(".docx"):
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif decoded_path.lower().endswith(".xlsx"):
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif decoded_path.lower().endswith(".pptx"):
            content_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

        # Get filename for download
        filename = decoded_path.split("/")[-1]

        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(content))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Download error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ralat memuat turun dokumen: {str(e)}")


@app.get("/api/documents")
async def list_documents(folder: str = "JKST", search: str = None):
    """
    List available documents in a folder with optional search.

    Query Parameters:
    - folder: Document folder (default: "JKST")
    - search: Search keyword (optional, searches in filename)

    Returns list of documents with download URLs.
    """
    try:
        client = get_storage_client()
        bucket = client.bucket(GCS_BUCKET_NAME)

        blobs = bucket.list_blobs(prefix=folder)

        documents = []
        for blob in blobs:
            if not blob.name.endswith("/"):  # Skip folders
                filename = blob.name.split("/")[-1]

                # Filter by search keyword if provided
                if search and search.lower() not in filename.lower():
                    continue

                # Determine file type
                file_ext = filename.split(".")[-1].lower() if "." in filename else "unknown"
                file_type_map = {
                    "pdf": "PDF",
                    "doc": "Word", "docx": "Word",
                    "xls": "Excel", "xlsx": "Excel",
                    "ppt": "PowerPoint", "pptx": "PowerPoint",
                    "txt": "Text",
                    "zip": "Archive", "rar": "Archive",
                    "jpg": "Image", "jpeg": "Image", "png": "Image", "gif": "Image"
                }
                file_type = file_type_map.get(file_ext, "Dokumen")

                documents.append({
                    "name": filename,
                    "path": blob.name,
                    "file_type": file_type,
                    "file_ext": file_ext,
                    "size_kb": round(blob.size / 1024, 1) if blob.size else 0,
                    "download_url": f"{BASE_URL}/api/download/{urllib.parse.quote(blob.name, safe='')}"
                })

        # Sort by name
        documents.sort(key=lambda x: x["name"])

        return {
            "folder": folder,
            "count": len(documents),
            "search": search,
            "documents": documents[:100]  # Limit to 100
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ralat: {str(e)}")


@app.get("/api/documents/search")
async def search_documents(query: str = ""):
    """
    Search for documents across all folders by keyword.

    Query Parameters:
    - query: Search keyword (required)

    Returns matching documents with download URLs.
    """
    try:
        if not query:
            raise HTTPException(status_code=400, detail="Sila sediakan keyword carian")

        client = get_storage_client()
        bucket = client.bucket(GCS_BUCKET_NAME)

        # Search across all blobs
        all_blobs = bucket.list_blobs()
        documents = []
        query_lower = query.lower()

        for blob in all_blobs:
            if blob.name.endswith("/"):  # Skip folders
                continue

            filename = blob.name.split("/")[-1]

            # Search in filename and path
            if query_lower in filename.lower() or query_lower in blob.name.lower():
                file_ext = filename.split(".")[-1].lower() if "." in filename else "unknown"
                file_type_map = {
                    "pdf": "PDF",
                    "doc": "Word", "docx": "Word",
                    "xls": "Excel", "xlsx": "Excel",
                    "ppt": "PowerPoint", "pptx": "PowerPoint",
                    "txt": "Text",
                    "zip": "Archive", "rar": "Archive",
                    "jpg": "Image", "jpeg": "Image", "png": "Image", "gif": "Image"
                }
                file_type = file_type_map.get(file_ext, "Dokumen")

                documents.append({
                    "name": filename,
                    "path": blob.name,
                    "folder": blob.name.split("/")[0] if "/" in blob.name else "JKST",
                    "file_type": file_type,
                    "file_ext": file_ext,
                    "size_kb": round(blob.size / 1024, 1) if blob.size else 0,
                    "download_url": f"{BASE_URL}/api/download/{urllib.parse.quote(blob.name, safe='')}"
                })

        # Sort by name, limit results
        documents.sort(key=lambda x: x["name"])

        return {
            "query": query,
            "count": len(documents),
            "documents": documents[:50]  # Limit to 50 results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ralat carian: {str(e)}")


# ============================================================================
# FEEDBACK SYSTEM
# ============================================================================
def ensure_feedback_csv_exists():
    """Ensure the feedback CSV file and directory exist with proper headers."""
    try:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.dirname(FEEDBACK_CSV_PATH)
        if logs_dir and not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)

        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(FEEDBACK_CSV_PATH):
            with open(FEEDBACK_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'message_id',
                    'user_question',
                    'ai_response',
                    'rating',
                    'comment',
                    'retrieval_sources_count',
                    'reference_documents'
                ])
    except Exception as e:
        print(f"Error ensuring feedback CSV exists: {e}")


def clean_csv_text(text: str, max_length: int = 1000) -> str:
    """Clean text for CSV: remove newlines, limit length, escape special chars."""
    if not text:
        return ""
    # Remove newlines and excess whitespace
    text = ' '.join(text.split())
    # Limit length
    text = text[:max_length]
    return text


def save_feedback_to_csv(feedback: FeedbackRequest):
    """Save feedback to CSV file with file locking to prevent concurrent write corruption."""
    try:
        ensure_feedback_csv_exists()

        # Clean text fields
        user_question = clean_csv_text(feedback.user_question)
        ai_response = clean_csv_text(feedback.ai_response)
        comment = clean_csv_text(feedback.comment) if feedback.comment else ""

        # Extract reference document names from retrieval_data
        ref_docs = ""
        if feedback.retrieval_data:
            doc_names = []
            for item in feedback.retrieval_data:
                if isinstance(item, dict) and 'filename' in item:
                    doc_names.append(item['filename'])
            ref_docs = '; '.join(doc_names[:5])  # Limit to 5 documents

        # Prepare row data
        row = [
            feedback.timestamp,
            feedback.message_id,
            user_question,
            ai_response,
            feedback.rating or "",
            comment,
            str(feedback.retrieval_sources) if feedback.retrieval_sources else "0",
            ref_docs
        ]

        # Write to CSV with file locking
        with open(FEEDBACK_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        print(f"Feedback saved for message {feedback.message_id}")

    except Exception as e:
        print(f"Error saving feedback to CSV: {e}")
        import traceback
        traceback.print_exc()


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback (rating and/or comment) for an AI response.

    The feedback is saved to a CSV file for analytics and future training.
    """
    try:
        # Validate rating value if present
        if feedback.rating and feedback.rating not in ['thumbs_up', 'thumbs_down']:
            raise HTTPException(
                status_code=400,
                detail="Rating mesti 'thumbs_up' atau 'thumbs_down'"
            )

        # Save feedback to CSV in background
        import threading
        thread = threading.Thread(target=save_feedback_to_csv, args=(feedback,))
        thread.daemon = True
        thread.start()

        return {
            "status": "ok",
            "message": "Terima kasih atas maklum balas anda!",
            "message_id": feedback.message_id
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in feedback endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Ralat menyimpan maklum balas: {str(e)}"
        )


@app.get("/api/feedback/stats")
async def feedback_stats():
    """Get feedback statistics."""
    try:
        ensure_feedback_csv_exists()

        total = 0
        thumbs_up = 0
        thumbs_down = 0
        with_comment = 0

        if os.path.exists(FEEDBACK_CSV_PATH):
            with open(FEEDBACK_CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row:
                        total += 1
                        if row.get('rating') == 'thumbs_up':
                            thumbs_up += 1
                        elif row.get('rating') == 'thumbs_down':
                            thumbs_down += 1
                        if row.get('comment'):
                            with_comment += 1

        return {
            "total_feedback": total,
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "with_comment": with_comment,
            "feedback_file": FEEDBACK_CSV_PATH
        }

    except Exception as e:
        print(f"Error getting feedback stats: {e}")
        return {
            "error": str(e),
            "total_feedback": 0
        }


# ============================================================================
# VOICE FEATURES ENDPOINTS (STT + TTS)
# ============================================================================

class TranscribeRequest(BaseModel):
    audio_data: str  # Base64 encoded audio


@app.post("/api/voice/transcribe")
async def transcribe_audio(req: TranscribeRequest):
    """
    Transcribe audio to text using OpenAI Whisper.
    Expects base64-encoded audio data.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key tidak dikonfigurasi"
        )

    try:
        client = get_openai_client()
        if not client:
            raise HTTPException(
                status_code=503,
                detail="OpenAI client tidak tersedia"
            )

        # Decode base64 audio
        audio_bytes = base64.b64decode(req.audio_data)

        # Create a file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.webm"  # Whisper needs a filename

        # Call Whisper API
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ms"  # Bahasa Melayu
        )

        return {
            "text": transcript.text,
            "language": "ms"
        }

    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Ralat transkripsi: {str(e)}"
        )


@app.post("/api/voice/synthesize")
async def synthesize_speech(request: Request):
    """
    Convert text to speech using Google Cloud TTS.
    Request body: {"text": "text to speak"}
    Returns: MP3 audio file
    """
    try:
        data = await request.json()
        text = data.get("text", "")

        if not text:
            raise HTTPException(status_code=400, detail="Teks kosong")

        # Limit text length (Google TTS has limits)
        if len(text) > 5000:
            text = text[:5000]

        # Generate speech
        audio_content = await asyncio.to_thread(
            synthesize_speech_google,
            text
        )

        return Response(
            content=audio_content,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=speech.mp3"
            }
        )

    except Exception as e:
        print(f"TTS error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Ralat TTS: {str(e)}"
        )


# ============================================================================
# LOCAL VOICE FEATURES (100% local STT + TTS for Ultra mode)
# ============================================================================

_whisper_model = None
_tts_model = None
_tts_tokenizer = None


def get_whisper_model():
    """Lazy-load faster-whisper model on GPU."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        model_size = os.environ.get("WHISPER_MODEL", "large-v3")
        device = os.environ.get("WHISPER_DEVICE", "cuda")
        compute = "float16" if device == "cuda" else "int8"
        print(f"[VOICE] Loading Whisper {model_size} on {device}...")
        _whisper_model = WhisperModel(model_size, device=device, compute_type=compute)
        print(f"[VOICE] Whisper ready on {device}")
    return _whisper_model


def get_tts_model():
    """Lazy-load Facebook MMS-TTS Malay model on GPU."""
    global _tts_model, _tts_tokenizer
    if _tts_model is None:
        import torch
        from transformers import VitsModel, AutoTokenizer
        model_name = os.environ.get("TTS_LOCAL_MODEL", "facebook/mms-tts-zlm")
        device = os.environ.get("TTS_LOCAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VOICE] Loading TTS model {model_name} on {device}...")
        _tts_model = VitsModel.from_pretrained(model_name).to(device)
        _tts_tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"[VOICE] TTS ready on {device}")
    return _tts_model, _tts_tokenizer


@app.post("/api/voice/transcribe/local")
async def transcribe_audio_local(req: TranscribeRequest):
    """
    Transcribe audio to text using local Faster-Whisper on GPU.
    100% local — no cloud API calls.
    """
    try:
        model = await asyncio.to_thread(get_whisper_model)

        # Decode base64 audio
        audio_bytes = base64.b64decode(req.audio_data)
        audio_file = io.BytesIO(audio_bytes)

        # Transcribe with Whisper
        segments, info = await asyncio.to_thread(
            model.transcribe, audio_file, language="ms", beam_size=5
        )
        segments_list = list(segments)
        text = " ".join(seg.text.strip() for seg in segments_list)

        return {
            "text": text,
            "language": info.language,
            "language_probability": round(info.language_probability, 2),
            "duration": round(info.duration, 1),
            "provider": "local-whisper"
        }

    except Exception as e:
        print(f"Local transcription error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Ralat transkripsi tempatan: {str(e)}"
        )


@app.post("/api/voice/synthesize/local")
async def synthesize_speech_local(request: Request):
    """
    Convert text to speech using local MMS-TTS (Facebook) on GPU.
    100% local — no cloud API calls. Returns WAV audio.
    """
    try:
        import torch
        import struct

        data = await request.json()
        text = data.get("text", "")

        if not text:
            raise HTTPException(status_code=400, detail="Teks kosong")

        # Limit text length
        if len(text) > 3000:
            text = text[:3000]

        def _synthesize(text_input):
            model, tokenizer = get_tts_model()
            device = next(model.parameters()).device
            inputs = tokenizer(text_input, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(**inputs).waveform
            # Convert to numpy
            waveform = output.squeeze().cpu().numpy()
            return waveform

        waveform = await asyncio.to_thread(_synthesize, text)

        # Convert numpy array to WAV bytes
        sample_rate = 22050  # MMS-TTS default
        audio_buf = io.BytesIO()
        # Write WAV header
        num_samples = len(waveform)
        data_size = num_samples * 2  # 16-bit = 2 bytes per sample
        audio_buf.write(b'RIFF')
        audio_buf.write(struct.pack('<I', 36 + data_size))
        audio_buf.write(b'WAVE')
        audio_buf.write(b'fmt ')
        audio_buf.write(struct.pack('<I', 16))  # chunk size
        audio_buf.write(struct.pack('<H', 1))   # PCM format
        audio_buf.write(struct.pack('<H', 1))   # mono
        audio_buf.write(struct.pack('<I', sample_rate))
        audio_buf.write(struct.pack('<I', sample_rate * 2))  # byte rate
        audio_buf.write(struct.pack('<H', 2))   # block align
        audio_buf.write(struct.pack('<H', 16))  # bits per sample
        audio_buf.write(b'data')
        audio_buf.write(struct.pack('<I', data_size))
        # Write audio data as 16-bit PCM
        import numpy as np
        pcm = (waveform * 32767).astype(np.int16)
        audio_buf.write(pcm.tobytes())

        audio_buf.seek(0)
        return Response(
            content=audio_buf.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=speech.wav"
            }
        )

    except Exception as e:
        print(f"Local TTS error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Ralat TTS tempatan: {str(e)}"
        )


# ============================================================================
# TELEGRAM BOT INTEGRATION
# ============================================================================

# Store conversation history per Telegram chat (in-memory, resets on restart)
telegram_conversations: Dict[int, List[Dict[str, str]]] = {}

# Maximum conversation history per chat
TELEGRAM_MAX_HISTORY = 10


async def send_telegram_message(chat_id: int, text: str, parse_mode: str = "HTML"):
    """Send a message to a Telegram chat."""
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not configured")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # Telegram has a 4096 character limit per message
    # Split long messages if needed
    max_length = 4000
    messages_to_send = []

    if len(text) > max_length:
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 < max_length:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    messages_to_send.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        if current_chunk:
            messages_to_send.append(current_chunk.strip())
    else:
        messages_to_send = [text]

    async with httpx.AsyncClient() as client:
        for msg in messages_to_send:
            payload = {
                "chat_id": chat_id,
                "text": msg,
                "parse_mode": parse_mode
            }
            try:
                response = await client.post(url, json=payload, timeout=30.0)
                if response.status_code != 200:
                    print(f"Telegram API error: {response.text}")
                    # Try without parse_mode if HTML parsing fails
                    if parse_mode == "HTML":
                        payload["parse_mode"] = None
                        payload["text"] = msg.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
                        await client.post(url, json=payload, timeout=30.0)
            except Exception as e:
                print(f"Error sending Telegram message: {e}")
                return False

    return True


async def send_telegram_typing(chat_id: int):
    """Send typing indicator to show the bot is processing."""
    if not TELEGRAM_BOT_TOKEN:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendChatAction"
    payload = {
        "chat_id": chat_id,
        "action": "typing"
    }

    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json=payload, timeout=10.0)
        except Exception as e:
            print(f"Error sending typing indicator: {e}")


def format_response_for_telegram(reply: str, retrieval: Optional[List] = None) -> str:
    """Format AI response for Telegram with proper formatting."""
    # Convert markdown to Telegram-compatible format
    formatted = reply

    # Convert **bold** to <b>bold</b>
    formatted = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', formatted)

    # Convert *italic* to <i>italic</i>
    formatted = re.sub(r'(?<![*])\*([^*\n]+?)\*(?![*])', r'<i>\1</i>', formatted)

    # Convert `code` to <code>code</code>
    formatted = re.sub(r'`([^`]+)`', r'<code>\1</code>', formatted)

    # Convert markdown links [text](url) to HTML <a> tags
    formatted = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', formatted)

    # Clean up any remaining markdown characters that might break HTML
    formatted = formatted.replace('###', '').replace('##', '').replace('#', '')

    # Add source references if available
    if retrieval and len(retrieval) > 0:
        formatted += "\n\n📚 <b>Rujukan:</b>"
        seen_sources = set()
        for i, source in enumerate(retrieval[:3], 1):  # Max 3 sources
            filename = source.get('filename', 'Dokumen')
            if filename not in seen_sources:
                seen_sources.add(filename)
                # Extract just the filename without path
                display_name = filename.split('/')[-1] if '/' in filename else filename
                formatted += f"\n{i}. {display_name}"

    return formatted


def get_telegram_conversation(chat_id: int) -> List[Dict[str, str]]:
    """Get conversation history for a Telegram chat."""
    if chat_id not in telegram_conversations:
        telegram_conversations[chat_id] = []
    return telegram_conversations[chat_id]


def add_to_telegram_conversation(chat_id: int, role: str, content: str):
    """Add a message to the conversation history."""
    if chat_id not in telegram_conversations:
        telegram_conversations[chat_id] = []

    telegram_conversations[chat_id].append({
        "role": role,
        "content": content
    })

    # Keep only the last N messages
    if len(telegram_conversations[chat_id]) > TELEGRAM_MAX_HISTORY * 2:
        telegram_conversations[chat_id] = telegram_conversations[chat_id][-TELEGRAM_MAX_HISTORY * 2:]


def clear_telegram_conversation(chat_id: int):
    """Clear conversation history for a chat."""
    if chat_id in telegram_conversations:
        telegram_conversations[chat_id] = []


@app.post("/api/telegram/webhook")
async def telegram_webhook(request: Request):
    """
    Webhook endpoint for Telegram Bot.

    This receives updates from Telegram when users send messages to your bot.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Telegram bot not configured. Set TELEGRAM_BOT_TOKEN environment variable."
        )

    try:
        update = await request.json()
        print(f"Telegram update received: {update}")

        # Handle message updates
        if "message" in update:
            message = update["message"]
            chat_id = message["chat"]["id"]
            text = message.get("text", "")
            user = message.get("from", {})
            username = user.get("username", user.get("first_name", "User"))

            # Handle commands
            if text.startswith("/"):
                command = text.split()[0].lower()

                if command == "/start":
                    welcome_msg = """🕌 <b>Salam Sejahtera!</b>

Saya adalah <b>Pembantu AI JKST</b> (Jabatan Kehakiman Syariah Terengganu).

Saya boleh membantu anda dengan:
• Prosedur mahkamah syariah
• Undang-undang keluarga Islam
• Perkhidmatan JKST
• Borang dan dokumen

<b>Arahan:</b>
/start - Mesej selamat datang
/clear - Padam sejarah perbualan
/help - Bantuan

Sila tanya soalan anda dalam Bahasa Melayu."""
                    await send_telegram_message(chat_id, welcome_msg)
                    return {"ok": True}

                elif command == "/clear":
                    clear_telegram_conversation(chat_id)
                    await send_telegram_message(chat_id, "✅ Sejarah perbualan telah dipadam.")
                    return {"ok": True}

                elif command == "/help":
                    help_msg = """📖 <b>Bantuan Pembantu AI JKST</b>

<b>Cara Menggunakan:</b>
Hanya taip soalan anda dan saya akan cuba membantu.

<b>Contoh Soalan:</b>
• Bagaimana nak memohon cerai?
• Apakah dokumen untuk perkahwinan?
• Di mana Mahkamah Syariah Kuala Terengganu?
• Waktu operasi JKST?

<b>Arahan:</b>
/start - Mesej selamat datang
/clear - Padam sejarah perbualan
/help - Paparan ini

<b>Nota:</b>
Jawapan adalah untuk panduan am sahaja. Sila rujuk pegawai JKST untuk maklumat rasmi.

📞 Hubungi: 09-623 2323
🌐 Web: syariah.terengganu.gov.my"""
                    await send_telegram_message(chat_id, help_msg)
                    return {"ok": True}

                else:
                    await send_telegram_message(chat_id, "❓ Arahan tidak dikenali. Taip /help untuk bantuan.")
                    return {"ok": True}

            # Process regular messages
            if text:
                # Send typing indicator
                await send_telegram_typing(chat_id)

                # Get conversation history
                conversation = get_telegram_conversation(chat_id)

                # Add user message to history
                add_to_telegram_conversation(chat_id, "user", text)

                # Build messages for AI
                messages_for_ai = conversation.copy()

                try:
                    # Classify query
                    query_type = classify_query(text)

                    # Retrieve from RAG
                    rag_contexts, rag_sources = await retrieve_from_rag(text)

                    # Search local knowledge
                    local_contexts, local_sources = await asyncio.to_thread(
                        search_local_knowledge, text
                    )

                    # Conditionally search web
                    web_contexts, web_sources = [], []
                    if query_type == "external" or query_type == "hybrid":
                        web_contexts, web_sources = await asyncio.to_thread(
                            search_web, text
                        )
                    elif query_type == "internal" and not rag_contexts and not local_contexts:
                        web_contexts, web_sources = await asyncio.to_thread(
                            search_web, text
                        )

                    # Combine sources
                    all_sources = rag_sources + local_sources + web_sources

                    # Generate response
                    reply = await asyncio.to_thread(
                        generate_with_gemini,
                        text,
                        rag_contexts,
                        web_contexts,
                        messages_for_ai,
                        query_type,
                        local_contexts
                    )

                    # Add assistant response to history
                    add_to_telegram_conversation(chat_id, "assistant", reply)

                    # Format for Telegram
                    formatted_reply = format_response_for_telegram(reply, all_sources)

                    # Send response
                    await send_telegram_message(chat_id, formatted_reply)

                except Exception as e:
                    print(f"Error processing Telegram message: {e}")
                    import traceback
                    traceback.print_exc()
                    error_msg = "⚠️ Maaf, ralat telah berlaku. Sila cuba semula atau hubungi JKST di 09-623 2323."
                    await send_telegram_message(chat_id, error_msg)

        return {"ok": True}

    except Exception as e:
        print(f"Telegram webhook error: {e}")
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


@app.get("/api/telegram/setup")
async def telegram_setup_webhook():
    """
    Setup the Telegram webhook.

    Call this endpoint once after deploying to register your webhook with Telegram.
    You need to set TELEGRAM_WEBHOOK_URL environment variable first.

    Example: https://your-domain.com/api/telegram/webhook
    """
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="TELEGRAM_BOT_TOKEN not configured"
        )

    if not TELEGRAM_WEBHOOK_URL:
        return {
            "status": "error",
            "message": "TELEGRAM_WEBHOOK_URL not configured. Set it to your public server URL.",
            "example": "https://your-domain.com"
        }

    webhook_url = f"{TELEGRAM_WEBHOOK_URL}/api/telegram/webhook"

    # Set webhook with Telegram
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"
    payload = {
        "url": webhook_url,
        "secret_token": TELEGRAM_WEBHOOK_SECRET,
        "allowed_updates": ["message"],
        "drop_pending_updates": True
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=30.0)
            result = response.json()

            if result.get("ok"):
                return {
                    "status": "success",
                    "message": "Webhook registered successfully",
                    "webhook_url": webhook_url,
                    "telegram_response": result
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to register webhook",
                    "telegram_response": result
                }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error setting webhook: {str(e)}"
            )


@app.get("/api/telegram/info")
async def telegram_bot_info():
    """Get information about the Telegram bot."""
    if not TELEGRAM_BOT_TOKEN:
        return {
            "configured": False,
            "message": "TELEGRAM_BOT_TOKEN not set"
        }

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)
            result = response.json()

            if result.get("ok"):
                bot_info = result.get("result", {})
                return {
                    "configured": True,
                    "bot_username": bot_info.get("username"),
                    "bot_name": bot_info.get("first_name"),
                    "bot_id": bot_info.get("id"),
                    "webhook_url": TELEGRAM_WEBHOOK_URL
                }
            else:
                return {
                    "configured": True,
                    "error": "Could not get bot info",
                    "telegram_response": result
                }
        except Exception as e:
            return {
                "configured": True,
                "error": str(e)
            }


@app.delete("/api/telegram/webhook")
async def telegram_delete_webhook():
    """Remove the Telegram webhook (switch to polling mode)."""
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="TELEGRAM_BOT_TOKEN not configured"
        )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, timeout=10.0)
            result = response.json()
            return {
                "status": "success" if result.get("ok") else "error",
                "telegram_response": result
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error deleting webhook: {str(e)}"
            )


# ============================================================================
# TELEGRAM POLLING MODE (no public IP / webhook needed)
# ============================================================================
TELEGRAM_POLLING = os.environ.get("TELEGRAM_POLLING", "true").lower() == "true"
_polling_offset = 0


async def telegram_poll_loop():
    """Background task: poll Telegram for new messages (no webhook needed)."""
    global _polling_offset

    if not TELEGRAM_BOT_TOKEN:
        return

    # First, delete any existing webhook so polling works
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook",
                timeout=10.0
            )
    except Exception:
        pass

    # Verify bot token works
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe",
                timeout=10.0
            )
            bot_info = resp.json()
            if bot_info.get("ok"):
                bot_name = bot_info["result"].get("username", "?")
                print(f"[TELEGRAM] Polling started for @{bot_name}")
            else:
                print(f"[TELEGRAM] Bot token invalid: {bot_info}")
                return
    except Exception as e:
        print(f"[TELEGRAM] Cannot reach Telegram API: {e}")
        return

    while True:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                    params={
                        "offset": _polling_offset,
                        "timeout": 30,
                        "allowed_updates": '["message"]'
                    },
                    timeout=35.0
                )
                data = resp.json()

            if data.get("ok") and data.get("result"):
                for update in data["result"]:
                    _polling_offset = update["update_id"] + 1

                    # Reuse the existing webhook handler logic
                    if "message" in update:
                        message = update["message"]
                        chat_id = message["chat"]["id"]
                        text = message.get("text", "")
                        user = message.get("from", {})
                        username = user.get("username", user.get("first_name", "User"))

                        if not text:
                            continue

                        print(f"[TELEGRAM] @{username}: {text[:80]}")

                        # Handle commands
                        if text.startswith("/"):
                            command = text.split()[0].lower()
                            if command == "/start":
                                welcome_msg = (
                                    "\U0001f54c <b>Salam Sejahtera!</b>\n\n"
                                    "Saya adalah <b>Pembantu AI JKST</b> (Jabatan Kehakiman Syariah Terengganu).\n\n"
                                    "Saya boleh membantu anda dengan:\n"
                                    "\u2022 Prosedur mahkamah syariah\n"
                                    "\u2022 Undang-undang keluarga Islam\n"
                                    "\u2022 Perkhidmatan JKST\n"
                                    "\u2022 Borang dan dokumen\n\n"
                                    "<b>Arahan:</b>\n"
                                    "/start - Mesej selamat datang\n"
                                    "/clear - Padam sejarah perbualan\n"
                                    "/help - Bantuan\n\n"
                                    "Sila tanya soalan anda dalam Bahasa Melayu."
                                )
                                await send_telegram_message(chat_id, welcome_msg)
                            elif command == "/clear":
                                clear_telegram_conversation(chat_id)
                                await send_telegram_message(chat_id, "\u2705 Sejarah perbualan telah dipadam.")
                            elif command == "/help":
                                help_msg = (
                                    "\U0001f4d6 <b>Bantuan Pembantu AI JKST</b>\n\n"
                                    "<b>Contoh Soalan:</b>\n"
                                    "\u2022 Bagaimana nak memohon cerai?\n"
                                    "\u2022 Apakah dokumen untuk perkahwinan?\n"
                                    "\u2022 Di mana Mahkamah Syariah Kuala Terengganu?\n\n"
                                    "\U0001f4de Hubungi: 09-623 2323\n"
                                    "\U0001f310 Web: syariah.terengganu.gov.my"
                                )
                                await send_telegram_message(chat_id, help_msg)
                            continue

                        # Process regular messages
                        await send_telegram_typing(chat_id)
                        conversation = get_telegram_conversation(chat_id)
                        add_to_telegram_conversation(chat_id, "user", text)

                        try:
                            # Use local/ultra mode for Telegram (works without Google Cloud)
                            result = await chat_local_or_ultra(
                                conversation.copy(), text, override_mode="ultra"
                            )
                            reply = result["reply"]
                            all_sources = result.get("retrieval", [])

                            add_to_telegram_conversation(chat_id, "assistant", reply)
                            formatted_reply = format_response_for_telegram(reply, all_sources)
                            await send_telegram_message(chat_id, formatted_reply)

                        except Exception as e:
                            print(f"[TELEGRAM] Error: {e}")
                            await send_telegram_message(
                                chat_id,
                                "\u26a0\ufe0f Maaf, ralat telah berlaku. Sila cuba semula."
                            )

        except httpx.ReadTimeout:
            # Normal — long polling timeout, just retry
            continue
        except Exception as e:
            print(f"[TELEGRAM] Polling error: {e}")
            await asyncio.sleep(5)


@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup."""
    if TELEGRAM_BOT_TOKEN and TELEGRAM_POLLING:
        asyncio.create_task(telegram_poll_loop())
