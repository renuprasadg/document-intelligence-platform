# GuardianRAG - Insurance Policy Q&A System

Production-grade RAG (Retrieval-Augmented Generation) system for UK insurance compliance.

## 🎯 Overview

GuardianRAG enables semantic search and question-answering over insurance policy documents with:
- ✅ Grounded answer generation (citations required)
- ✅ PII detection and redaction
- ✅ Complete audit trail for compliance
- ✅ Production-ready architecture

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key

### Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/renuprasadg/document-intelligence-platform.git
cd document-intelligence-platform
\`\`\`

2. Set up virtual environment:
\`\`\`bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -e ".[dev]"
\`\`\`

4. Configure environment:
\`\`\`bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
\`\`\`

5. Run the application:
\`\`\`bash
uvicorn knowledge_engine.main:app --reload
\`\`\`

Visit http://localhost:8000 to see the API.

## 📁 Project Structure

\`\`\`
src/knowledge_engine/
├── api/            # FastAPI routes
├── core/           # Configuration, logging
├── domain/         # Data models
├── services/       # Business logic
├── rag/            # RAG pipeline components
│   ├── cleaning/   # Document preprocessing
│   ├── chunking/   # Text segmentation
│   ├── embeddings/ # Vector generation
│   ├── retrieval/  # Document search
│   └── generation/ # Answer generation
└── utils/          # Utilities
\`\`\`

## 🧪 Testing

Run tests:
\`\`\`bash
pytest
\`\`\`

With coverage:
\`\`\`bash
pytest --cov=src --cov-report=html
\`\`\`

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Documentation](docs/api/)
- [Deployment Guide](docs/deployment/)

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details.

## 👤 Author

Renu Prasad G
- GitHub: [@renuprasadg](https://github.com/renuprasadg)

## 🙏 Acknowledgments

Built as part of GenAI/RAG learning curriculum focusing on production-ready systems.
# document-intelligence-platform