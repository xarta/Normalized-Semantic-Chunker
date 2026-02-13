![vLLM Remote Embeddings](https://img.shields.io/badge/vLLM-Remote%20Embeddings-green)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Lightweight](https://img.shields.io/badge/Image-274MB-blue)

![Normalized Semantic Chunker](logo.png)

# Normalized Semantic Chunker

## Forked 

Forked from https://github.com/smart-models/Normalized-Semantic-Chunker. We stripped out the in-container PyTorch/CUDA/sentence-transformers stack and replaced it with HTTP calls to our local vLLM embedding endpoint. The container no longer needs a GPU, CUDA runtime, or any ML frameworks — it's a lightweight FastAPI app (~274MB Docker image vs the original ~11GB+) that delegates all embedding work to a dedicated vLLM inference server.

## ⚠️ AI-Generated Content Notice

This project was **modified with AI assistance** and should be treated accordingly:

- **Not production-ready**: Created for a specific homelab environment.
- **May contain bugs**: AI-generated code can have subtle issues.
- **Author's Python experience**: The author (modifier) is not an experienced Python programmer.

### AI Tools Used

- GitHub Copilot (Claude models)
- Local vLLM instances for analysis and consolidation

### Licensing Note

Released under the **GNU GENERAL PUBLIC LICENSE Version 3**. Given the AI-generated/modified nature:
- The modifying author makes no claims about originality
- Use at your own risk
- If you discover any copyright concerns, please open an issue

---


## Modified README continuation: 

The Normalized Semantic Chunker is a cutting-edge tool that unlocks the full potential of semantic chunking in an expanded range of NLP applications processing text documents and splits them into semantically coherent segments while ensuring optimal chunk size for downstream NLP tasks.
This innovative solution builds upon concepts from [YouTube's Advanced Text Splitting for RAG](https://www.youtube.com/watch?v=8OJC21T2SL4&t=1930s) and implementation patterns from [LangChain's semantic chunker documentation](https://python.langchain.com/docs/how_to/semantic-chunker/).
Conventional semantic chunkers prioritize content coherence but often produce chunks with highly variable token counts. This leads to issues like context window overflow and inconsistent retrieval quality, significantly impacting token-sensitive applications such as retrieval-augmented generation (RAG).
The Normalized Semantic Chunker overcomes these challenges by combining semantic cohesion with statistical guarantees for token size compliance. It ensures chunks are not only semantically meaningful but also fall within an optimal size range in terms of token count. This enables more precise and efficient text preparation for embeddings, RAG pipelines, and other NLP applications.
Whether working with long documents, varied content structures, or token-sensitive NLP architectures, the Normalized Semantic Chunker provides a robust, adaptable solution for optimizing text segmentation.


## Key Features
-   **Adaptive Semantic Chunking**: Intelligently splits text based on semantic similarity between consecutive sentences.
-   **Precise Chunk Size Control**: Advanced algorithm statistically ensures compliance with maximum token limits.
-   **Parallel Multi-Percentile Optimization**: Efficiently searches for the optimal similarity percentile using parallel processing.
-   **Intelligent Small Chunk Management**: Automatically merges undersized chunks with their most semantically similar neighbors.
-   **Smart Oversized Chunk Handling**: Intelligently splits chunks that exceed token threshold limits while preserving semantic integrity.
-   **vLLM Remote Embeddings**: Delegates embedding generation to a remote vLLM endpoint via OpenAI-compatible API — no local GPU or ML frameworks required.
-   **Lightweight Container**: ~274MB Docker image (down from ~11GB with CUDA/PyTorch). Based on `python:3.11-slim`.
-   **Comprehensive Processing Pipeline**: From raw text to optimized chunks in a single workflow.
-   **Universal REST API with FastAPI**: Modern, high-performance API interface with automatic documentation, data validation, and seamless integration capabilities for any system or language.
-   **Docker Integration**: Easy deployment with Docker and docker-compose.
-   **Adaptive Processing**: Adjusts processing parameters based on document size for optimal resource usage.
-   **Format Support**: Handles text (.txt), markdown (.md), and structured JSON (.json) files.
-   **Resource Management**: Adjusts parallel processing workers based on document size.

## Table of Contents

- [How the Text Chunking Algorithm Works](#how-the-text-chunking-algorithm-works)
  - [The Pipeline](#the-pipeline)
  - [Statistical Control of Maximum Tokens Chunk Size](#statistical-control-of-maximum-tokens-chunk-size)
  - [Parallel Multi-Core Percentile Search Optimization](#parallel-multi-core-percentile-search-optimization)
  - [Comparison with Traditional Chunking](#comparison-with-traditional-chunking)
- [Advantages of the Solution](#advantages-of-the-solution)
  - [Optimal Preparation for RAG and Semantic Retrieval](#optimal-preparation-for-rag-and-semantic-retrieval)
  - [Superior Performance](#superior-performance)
  - [Flexibility and Customization](#flexibility-and-customization)
- [Installation and Deployment](#installation-and-deployment)
  - [Prerequisites](#prerequisites)
  - [Getting the Code](#getting-the-code)
  - [Local Installation with Uvicorn](#local-installation-with-uvicorn)
  - [Docker Deployment (Recommended)](#docker-deployment-recommended)
- [Using the API](#using-the-api)
  - [API Endpoints](#api-endpoints)
  - [Example API Call](#example-api-call)
  - [Response Format](#response-format)
- [Contributing](#contributing)

## How the Text Chunking Algorithm Works

### The Pipeline

The core innovation of Normalized Semantic Chunker lies in its multi-step pipeline that combines NLP techniques with statistical optimization to ensure both semantic coherence and size consistency:

1. The application exposes a simple REST API endpoint where users can upload a text document and parameters for maximum token limits and embedding model selection. 
2. The text is initially split into sentences using sophisticated regex pattern matching.
3. Each sentence is transformed into a vector embedding by calling a remote vLLM embedding endpoint (e.g. `Qwen/Qwen3-VL-Embedding-2B` — auto-detected from the server).
4. The angular similarity between consecutive sentence vectors is calculated.
5. A parallel search algorithm identifies the optimal percentile of the similarity distribution that respects the specified size constraints.
6. Chunks are formed by grouping sentences across boundaries identified by the chosen percentile.
7. A post-processing step identifies and merges chunks too small with their most semantically similar neighbours, ensuring size constraints are met.
8. A final step splits any remaining chunks that exceed the maximum token limit, prioritizing sentence boundaries.
9. The application returns a well-structured JSON response containing the chunks, metadata, and performance statistics, ready for immediate integration into production environments.

### Statistical Control of Maximum Tokens Chunk Size

Unlike traditional approaches, Normalized Semantic Chunker uses a sophisticated statistical method to ensure that chunks generally stay below a maximum token limit.

During the percentile search, potential chunkings are evaluated based on an estimate of their 95th percentile token count:

```python
# Calculate the estimated 95th percentile using z-score of 1.645
estimated_95th_percentile = average_tokens + (1.645 * std_dev)
if estimated_95th_percentile <= max_tokens:
    # This percentile is considered valid
    return chunks_with_tokens, percentile, average_tokens
```

This approach ensures that approximately 95% of the generated chunks respect the specified token limit while automatically handling the few edge cases through a subsequent splitting step.

### Parallel Multi-Core Percentile Search Optimization

The algorithm leverages parallel processing to simultaneously test multiple percentiles, significantly speeding up the search for the optimal splitting point:

```python
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(_process_percentile_range, args)
        for args in process_args
    ]
```

This parallel implementation allows for quickly finding the best balance between semantic cohesion and adherence to size constraints.

### Comparison with Traditional Chunking

| Feature | Traditional Chunking | Normalized Semantic Chunker |
|---------|----------------------|------------------------------|
| Boundary Determination | Fixed rules or token counts | Statistical analysis of semantic similarity distribution |
| Size Control | Often approximate or not guaranteed | Statistical guarantee (e.g., ~95%) + explicit splitting/merging |
| Semantic Cohesion | Can split related concepts | Preserves semantic cohesion via similarity analysis |
| Outlier Handling | Limited or absent | Intelligent merging of small chunks & splitting of large ones |
| Parallelization | Rarely implemented | Built-in parallel multi-core optimization |
| Adaptability | Requires manual parameter tuning | Automatically finds optimal parameters for each document type and size |

## Advantages of the Solution

### Optimal Preparation for RAG and Semantic Retrieval

Chunks generated by Normalized Semantic Chunker are ideal for Retrieval-Augmented Generation systems:

- **Semantic Coherence**: Each chunk contains semantically related information.
- **Balanced Sizes**: Chunks adhere to maximum size limits while avoiding excessively small fragments through merging.
- **Representativeness**: Each chunk aims to contain a complete and coherent unit of information.

### Superior Performance

The parallel implementation and statistical approach offer:

- **Processing Speed**: Parallel optimization on multi-core systems.
- **vLLM Remote Embeddings**: Embedding generation offloaded to a dedicated vLLM server — no local GPU required in the chunker container.
- **Scalability**: Efficient handling of large documents with adaptive processing based on document size.
- **Consistent Quality**: Predictable and reliable results regardless of text type.
- **Lightweight Deployment**: ~274MB Docker image with minimal dependencies.

### Flexibility and Customization

The algorithm adapts automatically to different types of content:

- **Adaptive Parameters**: Automatic identification of the best chunking parameters for each document.
- **Configurability**: Ability to specify custom maximum token limits (max_tokens) and control small chunk merging.
- **Extensibility**: Modular architecture easily extendable with new features.
- **Embedding Model Selection**: The embedding model is determined by what's loaded on your vLLM server.

## Installation and Deployment

### Prerequisites

- Docker and Docker Compose (for Docker deployment)
- A running vLLM embedding endpoint (e.g. on a separate GPU server)
- Python 3.11 (for local development)

### Getting the Code

Before proceeding with any installation method, clone the repository:
```bash
git clone https://github.com/smart-models/Normalized-Semantic-Chunker.git
cd Normalized-Semantic-Chunker
```

### Local Installation with Uvicorn

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   ```
   
   **For Windows users:**
   
   * Using Command Prompt:
   ```cmd
   .venv\Scripts\activate.bat
   ```
   
   * Using PowerShell:
   ```powershell
   # If you encounter execution policy restrictions, run this once per session:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   
   # Then activate the virtual environment:
   .venv\Scripts\Activate.ps1
   ```
   > **Note:** PowerShell's default security settings may prevent script execution. The above command temporarily allows scripts for the current session only, which is safer than changing system-wide settings.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn normalized_semantic_chunker:app --reload
   ```

4. The API will be available at `http://localhost:8000`.
   
   Access the API documentation and interactive testing interface at `http://localhost:8000/docs`.

### Docker Deployment (Recommended)

1. Create required directories for persistent storage:
   ```bash
   mkdir -p logs
   ```

2. Deploy with Docker Compose:

   ```bash
   cd docker
   docker compose up -d
   ```

   **Stopping the service**:
   ```bash
   docker compose down
   ```

   > **Note**: The container requires network access to the vLLM embedding endpoint. Configure `EMBEDDING_BASE_URL` and `EMBEDDING_API_KEY` in your environment or compose file.

3. The API will be available at `http://localhost:8000`.
   
   Access the API documentation and interactive testing interface at `http://localhost:8000/docs`.

## Using the API

### API Endpoints

- **POST `/normalized_semantic_chunker/`**  
  Chunks a text document into semantically coherent segments while controlling token size.
  
  **Parameters:**
  - `file`: The text file to be chunked (supports .txt, .md, and .json formats)
  - `max_tokens`: Maximum token count per chunk (integer, required)
  - `model`: Embedding model name (string, default: auto-detected from vLLM server)
  - `merge_small_chunks`: Whether to merge undersized chunks (boolean, default: `true`)
  - `verbosity`: Show detailed logs (boolean, default: `false`)
  
  **Response:**
  Returns a JSON object containing:
  - `chunks`: Array of text segments with their token counts and IDs
  - `metadata`: Processing statistics including chunk count, token statistics, percentile used, model name, and processing time

  **JSON Input Format:**
  When using JSON files as input, the expected structure is:
  ```json
  {
    "chunks": [
      {
        "text": "First chunk of text content...",
        "metadata_field": "Additional metadata is allowed..."
      },
      {
        "text": "Second chunk of text content...",
        "id": 12345
      },
      ...
    ]
  }
  ```
  The service will process each text chunk individually, maintaining the chunk boundaries provided in your JSON file, then apply semantic chunking within those boundaries as needed. Additional metadata fields beyond `text` are allowed and will be ignored during processing, so you can include any extra information you need while still having the JSON process correctly.

- **GET `/`**  
  Health check endpoint that returns service status, vLLM connectivity, embedding model, and API version.

### Example API Call using cURL

```bash
# Basic usage with required parameters
curl -X POST "http://localhost:8000/normalized_semantic_chunker/?max_tokens=512" \
  -F "file=@document.txt" 

# With all parameters specified
curl -X POST "http://localhost:8000/normalized_semantic_chunker/?max_tokens=512&merge_small_chunks=true&verbosity=false" \
  -F "file=@document.txt" \
  -H "accept: application/json"

# Health check endpoint
curl http://localhost:8000/
```

### Example API Call using Python

```python
import requests
import json

# Replace with your actual API endpoint if hosted elsewhere
api_url = 'http://localhost:8000/normalized_semantic_chunker/'
file_path = 'document.txt' # Your input text file
max_tokens_per_chunk = 512
merge_small_chunks = True  # Whether to merge undersized chunks with semantically similar neighbors
verbosity = False  # Whether to show detailed logs

try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/plain')}
        params = {
            'max_tokens': max_tokens_per_chunk,
            'merge_small_chunks': merge_small_chunks,
            'verbosity': verbosity
        }
        # if model_name: # Uncomment to specify a model (otherwise auto-detected from vLLM)
        #     params['model'] = model_name

        response = requests.post(api_url, files=files, params=params)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        result = response.json()

        print(f"Successfully chunked document into {result['metadata']['n_chunks']} chunks.")
        # Save the response to a file
        output_file = 'response.json'
        # print("Metadata:", result['metadata'])
        # print("First chunk:", result['chunks'][0])
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, indent=4, ensure_ascii=False)
        print(f"Response saved to {output_file}")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except requests.exceptions.RequestException as e:
    print(f"API Request failed: {e}")
    if e.response is not None:
        print("Error details:", e.response.text)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```


### Response Format

A successful chunking operation returns a `ChunkingResult` object:

```json
{
  "chunks": [
    {
      "text": "This is the first chunk of text...",
      "token_count": 480,
      "id": 1
    },
    {
      "text": "This is the second chunk...",
      "token_count": 505,
      "id": 2
    },
    {
      "text": "Additional chunks would appear here...",
      "token_count": 490,
      "id": 3
    }
  ],
  "metadata": {
    "n_chunks": 42,
    "avg_tokens": 495,
    "max_tokens": 510,
    "min_tokens": 150,
    "percentile": 85,
    "embedder_model": "Qwen/Qwen3-VL-Embedding-2B",
    "source": "your-document-source.txt",
    "processing_time": 15.78
  }
}
```


## Contributing

The Normalized Semantic Chunker is an open-source project that thrives on community contributions. Your involvement is not just welcome, it's fundamental to the project's growth, innovation, and long-term success.

Whether you're fixing bugs, improving documentation, adding new features, or sharing ideas, every contribution helps build a better tool for everyone. We believe in the power of collaborative development and welcome participants of all skill levels.

If you're interested in contributing:

1. Fork the repository
2. Create a development environment with all dependencies
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Submit a pull request

Happy Semantic Chunking!

## Ecosystem

This service is part of the [xarta](https://github.com/xarta) document analysis ecosystem — a set of Dockerised microservices built for a nested Proxmox homelab running local AI inference on an RTX 5090 and RTX 4000 Blackwell.

The project grew out of a practical need: AI-generated infrastructure code (Proxmox, LXC, Docker configs) is full of secrets and environment-specific details that make it unsafe to share and hard to reuse. These services clean, chunk, embed, and ingest that code into a vector database so AI agents can query it efficiently — and so sanitised versions can be published to GitHub.

All services delegate compute-heavy work (LLM chat, embeddings, reranking) to shared vLLM endpoints via OpenAI-compatible APIs, taking advantage of batched and parallel GPU operations rather than bundling models locally.

This is a work in progress, decomposing the original project that became too monolithic and difficult to develop into more manageable components.  The original project had many incomplete features that were proving difficult to implement with generative AI without impacting other features and so even when all components are migrated there will still be some development to do before the features originally envisaged are complete.

| Repository | Description |
|---|---|
| [Normalized-Semantic-Chunker](https://github.com/xarta/Normalized-Semantic-Chunker) | Embedding-based semantic text chunking with statistical token-size control. Lightweight fork — delegates embeddings to a remote vLLM endpoint. |
| [Agentic-Chunker](https://github.com/xarta/Agentic-Chunker) | LLM-driven proposition chunking — uses chat completions to semantically group content. Fork replacing Google Gemini with local vLLM. |
| [gitleaks-validator](https://github.com/xarta/gitleaks-validator) | Dockerised gitleaks wrapper — pattern-driven secret scanning and replacement via REST API. |
| [knowledge-service](https://github.com/xarta/knowledge-service) | Document ingestion into SeekDB (vector database) with RAG query interface. Composes chunking + embedding services. |
| [content-analyser](https://github.com/xarta/content-analyser) | Duplication detection and contradiction analysis across document sets. Composes chunking, embedding, and LLM services. |

---
