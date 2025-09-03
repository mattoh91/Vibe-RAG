# 🚀 Vibe RAG

A powerful full-stack RAG (Retrieval-Augmented Generation) application that brings your documents to life through intelligent conversation. Built with FastAPI backend and Streamlit frontend, supporting multiple LLM providers for maximum flexibility.

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)

## ✨ Features

### 🤖 **Multi-Provider LLM Support**
- **Azure OpenAI**: GPT-4o, GPT-4, GPT-3.5-turbo with custom deployments
- **OpenRouter**: Access to Claude Sonnet 4, GPT-5, Grok-4, DeepSeek, Gemini 2.5 Flash
- **Anthropic**: Direct Claude integration with latest models

### 📄 **Advanced Document Processing**
- **PDF Upload**: Drag-and-drop interface with real-time processing
- **Smart Chunking**: Hybrid recursive chunking optimized for long documents
- **Metadata Extraction**: Preserve document structure and context

### 🔍 **Intelligent Search & Retrieval**
- **Vector Search**: Powered by Qdrant with BGE embeddings
- **Semantic Reranking**: MS-MARCO cross-encoder for precision
- **Source Attribution**: Track and display document sources

### 🎨 **Modern User Experience**
- **Clean Interface**: Intuitive Streamlit UI with real-time feedback
- **Connection Testing**: Verify LLM configurations before use
- **System Monitoring**: Health checks and status indicators

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │     FastAPI     │    │     Qdrant      │
│   Frontend      │◄──►│    Backend      │◄──►│  Vector Store   │
│   (Port 12001)  │    │  (Port 12000)   │    │   (Embedded)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   LLM Providers │
                    │ Azure│OR│Claude  │
                    └─────────────────┘
```

### Backend Components
- **Document Service**: PDF processing with PyMuPDF
- **Embedding Service**: BGE-small-en-v1.5 for semantic understanding
- **Vector Service**: Qdrant integration for similarity search
- **Rerank Service**: Cross-encoder refinement
- **LLM Service**: Multi-provider abstraction layer

### Frontend Features
- **Document Management**: Upload, view, and manage PDF documents
- **LLM Configuration**: Dynamic provider setup with validation
- **Query Interface**: Natural language interaction with context
- **Real-time Updates**: Live status and progress indicators

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and start
git clone https://github.com/mattoh91/Vibe-RAG.git
cd Vibe-RAG
cp .env.example .env
docker-compose up -d

# Access applications
# Frontend: http://localhost:12001
# Backend API: http://localhost:12000/docs
```

### Option 2: Manual Setup

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 12000 --reload

# Frontend (new terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.port 12001 --server.address 0.0.0.0
```

## ⚙️ Configuration

### 🔑 LLM Provider Setup

#### Azure OpenAI
```
API Key: your-azure-openai-key
Endpoint: https://your-resource.openai.azure.com/
API Version: 2024-10-01-preview
Deployment Name: your-deployment-name
Model: gpt-4o (or your deployed model)
```

#### OpenRouter
```
API Key: your-openrouter-key
Model Options:
  - anthropic/claude-sonnet-4
  - openai/gpt-5
  - x-ai/grok-4
  - deepseek/deepseek-chat-v3.1
  - google/gemini-2.5-flash
```

#### Anthropic Claude
```
API Key: your-anthropic-key
Model Options:
  - claude-3-sonnet-20240229
  - claude-3-haiku-20240307
  - claude-3-opus-20240229
```

## 📚 Usage

1. **Start the Application**: Use Docker or manual setup
2. **Configure LLM**: Select provider and enter credentials in sidebar
3. **Upload Documents**: Drag PDF files to the upload area
4. **Ask Questions**: Query your documents in natural language
5. **Review Sources**: Check document sources and relevance scores

## 🛠️ Development

### Project Structure
```
Vibe-RAG/
├── backend/
│   ├── app/
│   │   ├── api/endpoints/     # API routes
│   │   ├── core/             # Configuration
│   │   ├── models/           # Data models
│   │   ├── schemas/          # Request/response schemas
│   │   └── services/         # Business logic
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit application
│   └── requirements.txt
├── docker-compose.yml       # Container orchestration
└── .env.example            # Environment template
```

### API Endpoints
- `GET /api/v1/health` - System health check
- `POST /api/v1/llm/configure` - Configure LLM provider
- `POST /api/v1/llm/test` - Test LLM connection
- `POST /api/v1/documents/upload` - Upload PDF document
- `GET /api/v1/documents/` - List uploaded documents
- `POST /api/v1/query/` - Query documents

## 🔧 Technical Details

### Chunking Strategy
- **Hybrid Recursive**: Preserves document structure
- **Overlap**: 200 characters for context continuity
- **Size**: 1000 characters optimal for embeddings

### Vector Search
- **Model**: BAAI/bge-small-en-v1.5 (384 dimensions)
- **Distance**: Cosine similarity
- **Index**: HNSW for fast retrieval

### Reranking
- **Model**: cross-encoder/ms-marco-MiniLM-L-12-v2
- **Purpose**: Refine search results for relevance
- **Threshold**: Configurable relevance scoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BGE Embeddings**: BAAI for high-quality embeddings
- **MS-MARCO**: Microsoft for cross-encoder models
- **Qdrant**: For efficient vector search capabilities
- **FastAPI**: For the robust backend framework
- **Streamlit**: For the intuitive frontend framework

---

**Built with ❤️ for intelligent document interaction**