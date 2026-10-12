#!/bin/bash

# Deploy documentation script for GenerativeRecommenders

set -e

echo "🚀 Deploying GenerativeRecommenders Documentation"

# Check if we're in the right directory
if [ ! -f "mkdocs.yml" ]; then
    echo "❌ Error: mkdocs.yml not found. Please run this script from the project root."
    exit 1
fi

# Install documentation dependencies if needed
if [ ! -f "venv/bin/activate" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

echo "📦 Installing documentation dependencies..."
source venv/bin/activate
pip install -r requirements-docs.txt

# Build the documentation
echo "🔨 Building documentation..."
mkdocs build --clean --strict

# Deploy to GitHub Pages (if on main branch)
if [ "$1" = "--deploy" ]; then
    echo "🚀 Deploying to GitHub Pages..."
    mkdocs gh-deploy --clean --message "Deploy documentation updates"
else
    echo "✅ Documentation built successfully!"
    echo "📁 Built files are in the 'site' directory"
    echo "🌐 To serve locally, run: mkdocs serve"
    echo "🚀 To deploy to GitHub Pages, run: $0 --deploy"
fi

echo "✨ Done!"