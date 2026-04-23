#!/usr/bin/env bash
# Build script for Render deployment

echo "🚀 Building Hallucination Detector..."

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

echo "✅ Build complete!"
