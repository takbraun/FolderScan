#!/bin/bash
cd /home/takbraun/nsfw-scanner
source venv/bin/activate
echo "🚀 Starting NSFW Scanner..."
echo "📍 Access at: http://localhost:8000"
echo "📊 Quarantine folder: /home/takbraun/nsfw-scanner/quarantine"
python3 app.py
