#!/bin/bash
# =============================================================================
# JKST Master AI — GX10 Performance-Tuned Setup Script
# Target: NVIDIA GB10 (Grace Blackwell) — aarch64
#   CPU:  20 ARM cores (10x Cortex-X925 + 10x Cortex-A725)
#   RAM:  122GB unified memory
#   GPU:  NVIDIA GB10, CUDA 13.0, unified memory with CPU
#   Disk: NVMe SSD
# =============================================================================

set -euo pipefail

PROJECT_DIR="/opt/jkst-ai"
FRONTEND_DIR="/var/www/jkst-ai/public"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_DIR/venv"
USER_NAME="$(whoami)"

echo "============================================"
echo "  JKST Master AI — GX10 Performance Setup"
echo "============================================"

# ---------- Platform detection ----------
echo ""
echo "[1/8] Detecting platform..."

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "WARNING: Expected aarch64 (ARM), got $ARCH. Proceeding anyway."
fi

GPU_NAME="none"
CUDA_VERSION="none"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' 2>/dev/null || echo "unknown")
    echo "  GPU: $GPU_NAME"
    echo "  CUDA: $CUDA_VERSION"
    echo "  Architecture: $ARCH"
    GPU=true
else
    echo "  No GPU detected — CPU-only mode"
    GPU=false
fi

CPU_CORES=$(nproc)
TOTAL_RAM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
echo "  CPU cores: $CPU_CORES"
echo "  RAM: ${TOTAL_RAM_GB}GB"

# Calculate optimal workers: for I/O-bound async app, 2*cores + 1
# But cap at sensible limit — each worker loads ML models into GPU memory
WORKERS=$(( CPU_CORES > 8 ? 8 : CPU_CORES ))
echo "  Uvicorn workers: $WORKERS (tuned for ML model memory sharing)"

# ---------- Create directories ----------
echo ""
echo "[2/8] Creating directories..."
sudo mkdir -p "$PROJECT_DIR"/{logs,knowledge,documents,chroma_db}
sudo mkdir -p "$FRONTEND_DIR"
sudo chown -R "$USER_NAME":"$USER_NAME" "$PROJECT_DIR"

# ---------- Copy files ----------
echo ""
echo "[3/8] Copying files..."
cp "$REPO_DIR"/backend/app.py "$PROJECT_DIR/"
cp "$REPO_DIR"/backend/providers.py "$PROJECT_DIR/"

sudo cp "$REPO_DIR"/frontend/*.html "$FRONTEND_DIR/"
sudo cp "$REPO_DIR"/frontend/*.png "$FRONTEND_DIR/" 2>/dev/null || true
sudo cp "$REPO_DIR"/frontend/*.jpg "$FRONTEND_DIR/" 2>/dev/null || true

cp "$REPO_DIR"/knowledge/* "$PROJECT_DIR/knowledge/"

# ---------- Python virtual environment ----------
echo ""
echo "[4/8] Creating Python virtual environment..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"

# Upgrade pip for wheel caching
pip install --upgrade pip wheel setuptools --quiet

# ---------- Install PyTorch (GPU vs CPU) ----------
echo ""
echo "[5/8] Installing PyTorch..."
if [ "$GPU" = true ]; then
    echo "  Installing GPU-accelerated PyTorch for aarch64 + CUDA..."
    # On Jetson/GX10 aarch64 with CUDA, pip install torch gets the right build
    # from PyPI which includes CUDA support for aarch64
    pip install --no-cache-dir torch torchvision 2>&1 | tail -5

    # Verify CUDA availability
    python3 -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
" || echo "  Warning: CUDA verification failed, will fall back to CPU"
else
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3
fi

# ---------- Install all dependencies ----------
echo ""
echo "[6/8] Installing Python dependencies..."
pip install --no-cache-dir \
    fastapi uvicorn[standard] httpx pydantic python-dotenv \
    google-genai google-cloud-aiplatform google-cloud-storage google-cloud-texttospeech google-auth \
    openai cohere tavily-python \
    chromadb sentence-transformers rank-bm25 \
    pymupdf python-docx beautifulsoup4 lxml \
    orjson \
    2>&1 | tail -10

echo "  Dependencies installed successfully"

# ---------- Pre-download ML models ----------
echo ""
echo "[7/8] Pre-downloading ML models (this may take a few minutes)..."

# Set HuggingFace cache to NVMe for fast loading
export HF_HOME="$PROJECT_DIR/.hf_cache"
mkdir -p "$HF_HOME"

python3 -c "
import os
os.environ['HF_HOME'] = '$HF_HOME'

print('  Downloading Malay embedding model...')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('mesolitica/mistral-embedding-191m-8k-contrastive')
# Warm up with a test encode
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
_ = model.encode(['test'], device=device)
print(f'  Embedding model ready on {device}')

print('  Downloading cross-encoder model...')
from sentence_transformers import CrossEncoder
ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
_ = ce.predict([('test query', 'test document')])
print('  Cross-encoder model ready')

print('  All models pre-downloaded and verified')
"

# ---------- Setup env, systemd, nginx ----------
echo ""
echo "[8/8] Configuring services..."

# Environment file
if [ ! -f "$PROJECT_DIR/backend.env" ]; then
    cp "$REPO_DIR/configs/backend.env.template" "$PROJECT_DIR/backend.env"

    # Inject GX10-tuned defaults
    cat >> "$PROJECT_DIR/backend.env" <<'ENVEOF'

# =============================================================================
# GX10 PERFORMANCE TUNING (auto-generated)
# =============================================================================
# RAG Mode: ultra gives best results on this hardware
RAG_MODE=ultra

# HuggingFace model cache on NVMe
HF_HOME=/opt/jkst-ai/.hf_cache

# GPU-accelerated embedding
EMBEDDING_DEVICE=cuda
CROSS_ENCODER_DEVICE=cuda

# Retrieval tuning (more docs = better recall, GPU handles the load)
LOCAL_RETRIEVAL_TOP_K=15
ULTRA_RETRIEVAL_TOP_K=20
RAG_INITIAL_RESULTS=20
RAG_FINAL_RESULTS=7

# Cache tuning (large RAM = longer cache)
CACHE_TTL_SECONDS=600

# ChromaDB on NVMe
CHROMA_PERSIST_DIR=/opt/jkst-ai/chroma_db

# Ingestion tuning
INGEST_BATCH_SIZE=500
EMBEDDING_BATCH_SIZE=256

# Paths
LOCAL_KNOWLEDGE_PATH=/opt/jkst-ai/knowledge
LOCAL_DOCUMENTS_PATH=/opt/jkst-ai/documents
CSV_LOG_PATH=/opt/jkst-ai/logs/conversations.csv
FEEDBACK_CSV_PATH=/opt/jkst-ai/logs/feedback.csv
ENVEOF

    echo "  Created backend.env with GX10-tuned defaults"
    echo ""
    echo "  >>> IMPORTANT: Edit $PROJECT_DIR/backend.env and add your API keys <<<"
    echo ""
fi

# systemd service (use repo tuned version)
echo "  Installing systemd service..."
# Generate service file with correct user and worker count
cat > /tmp/jkst-ai.service <<SVCEOF
[Unit]
Description=JKST AI Assistant backend (FastAPI + uvicorn) — GX10 Tuned
After=network.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=3

[Service]
Type=simple
User=$USER_NAME
Group=$USER_NAME
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$PROJECT_DIR/backend.env

# GX10: $WORKERS workers, uvloop for async perf, HTTP/1.1 for proxy compat
ExecStart=$VENV/bin/uvicorn app:app \
    --host 127.0.0.1 \
    --port 8001 \
    --workers $WORKERS \
    --loop uvloop \
    --http h11 \
    --limit-concurrency 200 \
    --timeout-keep-alive 65 \
    --backlog 2048 \
    --no-access-log

# Restart configuration
Restart=always
RestartSec=3

# GX10 Resource limits — generous for 122GB RAM + GPU
MemoryMax=16G
MemoryHigh=12G
CPUQuota=90%

# GPU access
SupplementaryGroups=video render

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$PROJECT_DIR/logs $PROJECT_DIR/chroma_db $PROJECT_DIR/.hf_cache

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=jkst-assistant

[Install]
WantedBy=multi-user.target
SVCEOF

sudo cp /tmp/jkst-ai.service /etc/systemd/system/
sudo systemctl daemon-reload
echo "  systemd service installed (User=$USER_NAME, Workers=$WORKERS)"

# nginx
if command -v nginx &>/dev/null; then
    echo "  Installing nginx config..."
    sudo cp "$REPO_DIR/configs/jkst-ai.conf" /etc/nginx/sites-enabled/
    sudo nginx -t 2>&1 && sudo systemctl reload nginx
    echo "  nginx configured"
fi

# ---------- Summary ----------
echo ""
echo "============================================"
echo "  GX10 Performance Setup Complete!"
echo "============================================"
echo ""
echo "  Hardware Profile:"
echo "    CPU:     $CPU_CORES cores ($ARCH)"
echo "    RAM:     ${TOTAL_RAM_GB}GB"
echo "    GPU:     $GPU_NAME (CUDA $CUDA_VERSION)"
echo "    Workers: $WORKERS uvicorn processes"
echo ""
echo "  Tuning Applied:"
echo "    - GPU-accelerated PyTorch for embeddings + reranking"
echo "    - uvloop async event loop"
echo "    - $WORKERS workers with 200 concurrent connections"
echo "    - 16GB memory limit (headroom for ML models)"
echo "    - HuggingFace models pre-cached on NVMe"
echo "    - ChromaDB on NVMe for fast vector search"
echo ""
echo "  Next steps:"
echo "    1. Edit $PROJECT_DIR/backend.env with your API keys"
echo "    2. Put documents in $PROJECT_DIR/documents/"
echo "    3. Run ingestion: ./scripts/ingest.sh"
echo "    4. Start: sudo systemctl start jkst-ai"
echo "    5. Check: curl http://localhost:8001/api/health"
echo ""
