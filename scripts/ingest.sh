#!/bin/bash
# Re-ingest all documents into ChromaDB
# Run this after adding new documents to /opt/jkst-master-ai/documents/

set -e

PROJECT_DIR="/opt/jkst-master-ai"

echo "Clearing existing ChromaDB..."
rm -rf $PROJECT_DIR/chroma_db
mkdir -p $PROJECT_DIR/chroma_db

echo "Starting ingestion..."
cd $PROJECT_DIR
$PROJECT_DIR/venv/bin/python3 -c "
import os, sys
os.environ['RAG_MODE'] = 'local'
sys.path.insert(0, '.')
from providers import ingest_knowledge_to_chroma, get_chroma_collection
print('Ingesting all documents...')
ingest_knowledge_to_chroma()
collection = get_chroma_collection()
print(f'Done! ChromaDB has {collection.count()} chunks')
"

echo ""
echo "Ingestion complete. Restart the service:"
echo "  sudo systemctl restart jkst-master-ai"
