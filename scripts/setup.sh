#!/bin/bash
# JKST Master AI — Setup Script
# Works on both regular Ubuntu and NVIDIA DGX/GX10

set -e

PROJECT_DIR="/opt/jkst-master-ai"
FRONTEND_DIR="/var/www/jkst-master-ai/public"

echo "============================================"
echo "  JKST Master AI — Setup"
echo "============================================"

# Detect platform
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
    GPU=true
else
    echo "No GPU detected — CPU-only mode"
    GPU=false
fi

# Create directories
echo "Creating directories..."
sudo mkdir -p $PROJECT_DIR/{logs,knowledge,documents,chroma_db}
sudo mkdir -p $FRONTEND_DIR
sudo chown -R $USER:$USER $PROJECT_DIR

# Copy files
echo "Copying backend..."
cp backend/app.py $PROJECT_DIR/
cp backend/providers.py $PROJECT_DIR/

echo "Copying frontend..."
sudo cp frontend/*.html $FRONTEND_DIR/
sudo cp frontend/*.png $FRONTEND_DIR/ 2>/dev/null || true
sudo cp frontend/*.jpg $FRONTEND_DIR/ 2>/dev/null || true

echo "Copying knowledge files..."
cp knowledge/* $PROJECT_DIR/knowledge/

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv $PROJECT_DIR/venv
source $PROJECT_DIR/venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
if [ "$GPU" = true ]; then
    pip install --no-cache-dir torch
else
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
fi

pip install --no-cache-dir \
    fastapi uvicorn httpx pydantic \
    google-genai google-cloud-aiplatform google-cloud-storage google-cloud-texttospeech google-auth \
    openai cohere tavily-python \
    chromadb sentence-transformers rank-bm25 \
    pymupdf python-docx

# Setup env file
if [ ! -f $PROJECT_DIR/backend.env ]; then
    echo "Creating backend.env from template..."
    cp configs/backend.env.template $PROJECT_DIR/backend.env
    echo ""
    echo ">>> IMPORTANT: Edit $PROJECT_DIR/backend.env and add your API keys <<<"
    echo ""
fi

# Setup systemd service
echo "Installing systemd service..."
sudo cp configs/jkst-master-ai.service /etc/systemd/system/
sudo systemctl daemon-reload

# Setup nginx (optional)
if command -v nginx &>/dev/null; then
    echo "Installing nginx config..."
    sudo cp configs/jkst-master-ai.conf /etc/nginx/sites-enabled/
    sudo nginx -t && sudo systemctl reload nginx
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit $PROJECT_DIR/backend.env with your API keys"
echo "  2. Put documents in $PROJECT_DIR/documents/"
echo "  3. Set RAG_MODE in backend.env (google/local/ultra)"
echo "  4. Start: sudo systemctl start jkst-master-ai"
echo "  5. Check: curl http://localhost:8001/api/health"
echo ""
