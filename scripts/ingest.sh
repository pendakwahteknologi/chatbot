#!/bin/bash
# =============================================================================
# JKST Master AI — GX10 GPU-Accelerated Document Ingestion
# Re-ingest all documents into ChromaDB using GPU embedding
# Run this after adding new documents to /opt/jkst-ai/documents/
# =============================================================================

set -e

PROJECT_DIR="/opt/jkst-ai"

echo "============================================"
echo "  JKST Master AI — Document Ingestion"
echo "============================================"

# Check GPU availability
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "GPU: $GPU_NAME"
else
    echo "No GPU detected — will use CPU (slower)"
fi

# Count documents
DOC_COUNT=$(find "$PROJECT_DIR/knowledge" "$PROJECT_DIR/documents" -type f \( -name '*.md' -o -name '*.txt' -o -name '*.pdf' -o -name '*.docx' \) 2>/dev/null | wc -l)
echo "Documents found: $DOC_COUNT"
echo ""

# Clear existing ChromaDB
echo "Clearing existing ChromaDB..."
rm -rf "$PROJECT_DIR/chroma_db"
mkdir -p "$PROJECT_DIR/chroma_db"

echo "Starting GPU-accelerated ingestion..."
echo ""

cd "$PROJECT_DIR"

# Set environment for GPU acceleration
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-256}"
export INGEST_BATCH_SIZE="${INGEST_BATCH_SIZE:-500}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
export RAG_MODE=local

time $PROJECT_DIR/venv/bin/python3 -c "
import os, sys, time
sys.path.insert(0, '.')

# Load env file if exists
env_file = '$PROJECT_DIR/backend.env'
if os.path.exists(env_file):
    from dotenv import load_dotenv
    load_dotenv(env_file)

from providers import ingest_knowledge_to_chroma, get_chroma_collection, DEVICE

print(f'Embedding device: {DEVICE}')
print(f'Batch size: {os.environ.get(\"EMBEDDING_BATCH_SIZE\", \"256\")}')
print()

t0 = time.time()
ingest_knowledge_to_chroma()
elapsed = time.time() - t0

collection = get_chroma_collection()
count = collection.count()
print()
print(f'=== Ingestion Complete ===')
print(f'ChromaDB chunks: {count}')
print(f'Total time: {elapsed:.1f}s')
if count > 0:
    print(f'Speed: {count/elapsed:.0f} chunks/s')
"

echo ""
echo "============================================"
echo "  Ingestion complete!"
echo "============================================"
echo ""
echo "Restart the service to use new data:"
echo "  sudo systemctl restart jkst-ai"
