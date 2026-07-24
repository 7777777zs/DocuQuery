# DocuQuery

DocuQuery is a small RAG-based document Q&A API. It accepts PDF files, indexes their extracted text in a process-local FAISS store, and answers questions with source citations.

## Setup

The project targets Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell, activate the virtual environment with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. The remaining settings have useful defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `CHUNK_SIZE` | `1000` | Maximum chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `TOP_K` | `5` | Retrieved chunks per query |

## Run

```bash
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive OpenAPI documentation is available at `/docs`.

In a second terminal, start the Streamlit frontend on a separate port:

```bash
streamlit run frontend.py --server.port 8501
```

Open `http://localhost:8501`, upload a PDF, and ask questions after the API has indexed it.

## API

Upload a PDF:

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@./example.pdf"
```

Ask a question using the returned `document_id`:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the document about?","document_id":"<document-id>"}'
```

The process-local document store is cleared whenever the API restarts.

## Test

```bash
pytest
```

The route tests mock ingestion, retrieval, and generation boundaries, so they do not require OpenAI network access or API usage.
