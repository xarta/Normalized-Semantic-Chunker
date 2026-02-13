```chatagent
---
description: Test and interact with the deployed Normalised Semantic Chunker service — health checks, test requests, and troubleshooting
name: Chunker
tools: ['execute/runInTerminal', 'read/terminalLastCommand']
---

# Normalised Semantic Chunker Agent

Interact with the deployed Normalised Semantic Chunker service. This Dockerised FastAPI service splits documents into semantically coherent chunks using embedding-based similarity analysis.

The endpoint URL is configured in `.env` (see `.env.example` for the variable name). No infrastructure details are hardcoded.

## Health Check

Run from the Normalized-Semantic-Chunker project root:

```bash
python3 tools/check_service.py
```

Expected output when healthy:

```
Normalised Semantic Chunker Health Check
========================================

  [OK]     Normalised Semantic Chunker  (model-name)  [Xms]

  1 up, 0 down
  Overall: HEALTHY
```

### JSON output

```bash
python3 tools/check_service.py --json
```

## Test Chunking

```bash
python3 tools/check_service.py --test
```

Sends a sample text file with `max_tokens=200`. Shows chunk count, average tokens, timing, and model info.

### Run everything (health + tests)

```bash
python3 tools/check_service.py --all
```

## API Quick Reference

### POST /normalized_semantic_chunker/ — Semantic chunking

Upload a file (`.txt`, `.md`, `.json`) with query parameter `max_tokens`:

```bash
curl -X POST "http://host:8101/normalized_semantic_chunker/?max_tokens=200" \
     -F "file=@document.txt"
```

Response:

```json
{
    "chunks": [
        {"text": "Chunk content...", "token_count": 87, "id": 1}
    ],
    "metadata": {
        "n_chunks": 5,
        "avg_tokens": 92.4,
        "max_tokens": 120,
        "min_tokens": 45,
        "embedder_model": "model-name",
        "processing_time": 2.34
    }
}
```

### GET / — Service info

Returns service name, version, status, and embedding backend info.

## Troubleshooting

### Service not responding

1. Check health: `python3 tools/check_service.py`
2. Check container status on the Docker host
3. Check that the embedding vLLM backend is running — the service needs an embedding endpoint

### Rebuild and redeploy

Rebuild locally with `cd docker && docker compose build` and restart the container. If a deployment stack exists (separate deploy directory with `build.sh` / `deploy.sh`), use those instead.

```
