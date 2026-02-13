# Normalised Semantic Chunker — GitHub Copilot Instructions

## Project Overview

Normalised Semantic Chunker is a Dockerised FastAPI service that splits documents into semantically coherent chunks using embedding-based similarity analysis. It computes sentence embeddings, identifies semantic boundaries via normalised similarity scoring, and produces size-controlled chunks.

Originally forked from a local-model implementation, this version delegates all embedding work to a remote vLLM endpoint (OpenAI-compatible API) so no local GPU is required in the container.

## Key Rules

- **Use `python3`** not `python`.
- **British spelling** — `sanitise`, `analyse`, `colour`, etc.
- **No real infrastructure in source** — never put real IPs, hostnames, LXC IDs, or API keys in committed code. All loaded from `.env` (gitignored) or container environment variables.

## Project Structure

```
Normalized-Semantic-Chunker/
├── normalized_semantic_chunker.py   # FastAPI app + chunking logic
├── requirements.txt                 # Python deps
├── LICENSE
├── README.md
├── WHAT_IS_IT.md                    # Algorithm explanation
├── VLLM-MIGRATION-PLAN.md          # Remote embedding migration notes
├── .env                             # Real endpoint/secrets (GITIGNORED — never commit)
├── .env.example                     # Template showing variable names
├── .gitignore
├── .github/
│   ├── copilot-instructions.md      # This file
│   ├── agents/
│   │   └── chunker.agent.md         # @chunker agent for health/test
│   └── workflows/
│       └── docker-publish.yml       # CI workflow
├── tools/
│   └── check_service.py             # Health check + test tool
├── docker/
│   ├── dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt             # Docker-specific deps
└── test/
    ├── pytest.ini
    ├── test_api.py
    └── test_data/
        ├── alice_in_wonderland.txt
        ├── request.py
        └── response.json
```

## Environment Variables

The service container gets its config from environment variables injected at deploy time:

| Variable | Purpose |
|----------|---------|
| `EMBEDDING_BASE_URL` | vLLM OpenAI-compatible embedding endpoint |
| `EMBEDDING_API_KEY` | vLLM Bearer token |
| `EMBEDDING_MODEL_NAME` | Model name (empty = auto-detect) |
| `EMBEDDER_MODEL` | Alternative model name variable (default: `auto`) |
| `MAX_FILE_SIZE` | Maximum upload file size in bytes (default: 10MB) |
| `MAX_WORKERS` | Max parallel workers (default: CPU count - 1, max 4) |
| `CACHE_TIMEOUT` | Embedding cache timeout in seconds (default: 3600) |
| `EMBEDDING_TIMEOUT` | HTTP timeout for embedding calls (default: 120s) |

The client-side `.env` configures the tools for interacting with the deployed service:

| Variable | Purpose |
|----------|---------|
| `NORMALISED_CHUNKER_URL` | URL of the deployed service (e.g. `http://host:8101`) |

## API Endpoints

### GET / — Service info
Returns service name, version, status, and embedding backend details.

### POST /normalized_semantic_chunker/ — Chunk a document
Accepts a file upload (`.txt`, `.md`, `.json`) and returns semantically chunked output.

**Query parameters:**
- `max_tokens` (required) — Maximum tokens per chunk
- `model` (optional, default: `auto`) — Embedding model name
- `merge_small_chunks` (optional, default: `true`) — Merge undersized chunks
- `verbosity` (optional, default: `false`) — Verbose server-side logging

## Running Locally

```bash
# Set environment variables
export EMBEDDING_BASE_URL=http://your-vllm:8000/v1
export EMBEDDING_API_KEY=your-key

# Install deps
pip install -r requirements.txt

# Run
uvicorn normalized_semantic_chunker:app --host 0.0.0.0 --port 8101
```

## Health Check

```bash
python3 tools/check_service.py              # Health check
python3 tools/check_service.py --test       # Test chunking
python3 tools/check_service.py --all        # Everything
python3 tools/check_service.py --json       # JSON output
```

## Build & Deploy

The service is built as a Docker image. For deployment:

```bash
cd docker
docker compose build
```

If a deployment stack exists (e.g. `build.sh` / `deploy.sh` in a separate deploy directory), use those scripts to push to a local registry and deploy to the Docker host.

## Architecture Notes

- All embedding work is delegated to a remote vLLM endpoint — no local GPU required in the container.
- Uses normalised similarity scoring to detect semantic boundaries between sentences.
- Supports parallel processing for large documents with automatic worker scaling based on document size and available memory.
- Much faster than LLM-driven chunking approaches (embeddings only, no LLM calls) but less semantically precise.
- Best suited for large documents and batch processing.
