# NotebookLM Clone

A powerful NotebookLM clone that uses **Qdrant** for vector storage and **Mistral** for AI generation, optimized for cloud deployment.

## Key Features

- ** Multi-format Document Processing**: PDFs, text files, audio, YouTube videos, websites
- ** Advanced RAG**: Retrieval-Augmented Generation with proper citations
- ** Interactive Chat**: Ask questions about your documents with source citations
- ** Podcast Generation**: Create AI-generated podcast discussions from your content
- ** Smart Search**: Semantic search across all your documents
- ** Memory Layer**: Conversation history with Zep integration
- ** Cloud-Optimized**: Uses Qdrant cloud for fast, scalable vector storage

## Technology Stack

- **Vector Database**: Qdrant Cloud (fast, scalable, cloud-optimized)
- **AI Model**: Mistral (powerful, cost-effective language model)
- **Embeddings**: BAAI/bge-small-en-v1.5
- **Web Framework**: Streamlit
- **Audio Processing**: AssemblyAI
- **Web Scraping**: Firecrawl
- **Memory**: Zep Cloud

## Prerequisites

1. **Mistral API** - Get from [console.mistral.ai](https://console.mistral.ai/)
2. **Qdrant Cloud** - Get from [cloud.qdrant.io](https://cloud.qdrant.io/)
3. **AssemblyAI** - Get from [assemblyai.com](https://www.assemblyai.com/)
4. **Firecrawl** - Get from [firecrawl.dev](https://www.firecrawl.dev/)
5. **Zep Cloud** - Get from [getzep.com](https://www.getzep.com/)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/princ0301/Knowledger.git
   cd Knowledger
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables** (create `.env` file):
   ```env
   MISTRAL_API_KEY="your_mistral_api_key"
   QDRANT_API_KEY="your_qdrant_api_key"
   QDRANT_CLUSTER="your_qdrant_cluster_url"
   ASSEMBLYAI_API_KEY="your_assemblyai_key"
   FIRECRAWL_API_KEY="your_firecrawl_key"
   ZEP_API_KEY="your_zep_key"
   ```

## Running the App

### Option 1: Using the run script
```bash
uv run python run_app.py
```

### Option 2: Direct Streamlit command
```bash
uv run streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Testing the Pipeline

Test the core functionality:
```bash
uv run -m test.notebook_pipeline
```

## Usage

1. **Start the app** and open `http://localhost:8501`
2. **Add API keys** in the sidebar
3. **Upload documents** in the "Add Sources" tab:
   - PDFs, text files, audio files
   - YouTube video URLs
   - Website URLs
   - Paste text directly
4. **Chat with your documents** in the "Chat" tab
5. **Generate podcasts** in the "Studio" tab

## Project Structure

```
Knowledger/
├── src/
│   ├── doc/                    # Document processing
│   ├── embeddings/             # Embedding generation
│   ├── vector_db/              # Vector database (Qdrant)
│   ├── generation/             # RAG with Mistral
│   ├── memory/                 # Conversation memory
│   ├── audio_processing/       # Audio transcription
│   ├── web_scraping/          # Web content extraction
│   └── podcast/               # Podcast generation
├── test/                      # Test files
├── app.py                     # Streamlit web app
├── run_app.py                 # App runner script
└── README.md                  # This file
```
