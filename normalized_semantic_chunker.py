import os
import time
import json
import logging
import re
import psutil
import tiktoken
import multiprocessing
import numpy as np
import httpx
from logging.handlers import RotatingFileHandler
from typing import List, Optional, Dict, Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel, Field
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from contextlib import asynccontextmanager


RESERVED_SYSTEM_MEMORY_GB = 4.0  # GB of RAM to keep free for the OS and other processes
BASE_MEMORY_PER_WORKER_GB = 1.0  # Base GB of RAM allocated per worker
MEMORY_PER_SENTENCE_GB = 0.0004  # Additional GB of RAM per sentence per worker
DOC_SIZE_FACTOR_SCALER = (
    3000  # Scaler used in document size factor for CPU-based worker calculation
)
VERY_LARGE_DOC_SENTENCE_THRESHOLD = (
    20000  # Sentences to be considered a very large document
)
LARGE_DOC_SENTENCE_THRESHOLD = 10000  # Sentences to be considered a large document
MIN_SENTENCES_FOR_PARALLEL = (
    100  # Minimum sentences to use parallel processing instead of sequential
)
WORKERS_VERY_LARGE_DOC = 1  # Max workers for very large documents
WORKERS_LARGE_DOC = 2  # Max workers for large documents
STEP_SIZE_VERY_LARGE_DOC_THRESHOLD = (
    15000  # Sentence count threshold for using smallest step size
)
STEP_SIZE_LARGE_DOC_THRESHOLD = (
    5000  # Sentence count threshold for using medium step size
)
STEP_SIZE_DEFAULT = 10  # Default step size for smaller documents
STEP_SIZE_LARGE_DOC = 5  # Step size for large documents
STEP_SIZE_VERY_LARGE_DOC = 3  # Step size for very large documents

ALLOWED_EXTENSIONS = {"txt", "md", "json"}
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "auto")
MAX_FILE_SIZE = int(
    os.environ.get("MAX_FILE_SIZE", 10 * 1024 * 1024)
)  # 10MB di default
MAX_WORKERS = int(
    os.environ.get("MAX_WORKERS", min(multiprocessing.cpu_count() - 1, 4))
)
CACHE_TIMEOUT = int(os.environ.get("CACHE_TIMEOUT", 3600))  # 1 ora in secondi

# Remote embedding endpoint configuration
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "")  # auto-detect if empty
EMBEDDING_TIMEOUT = int(os.environ.get("EMBEDDING_TIMEOUT", 120))
EMBEDDING_MAX_RETRIES = int(os.environ.get("EMBEDDING_MAX_RETRIES", 3))
EMBEDDING_RETRY_BASE_DELAY = float(os.environ.get("EMBEDDING_RETRY_BASE_DELAY", 1.0))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Using remote embedding backend (vLLM)")
    try:
        model_name = _detect_vllm_model()
        logger.info(f"Remote embedding model detected: {model_name}")
    except Exception as e:
        logger.error(f"Failed to connect to remote embedding endpoint: {str(e)}")

    yield

    # Cleanup
    logger.info("Application shutting down, cleaning up resources...")


app = FastAPI(
    title="Normalized Semantic Chunker",
    description="API for processing and chunking text documents into smaller, semantically coherent segments",
    version="0.8.0",
    lifespan=lifespan,
)

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging (this sets up root logger with a console handler)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get the logger
logger = logging.getLogger(__name__)

# Create a file handler for error logs
error_log_path = logs_dir / "errors.log"
file_handler = RotatingFileHandler(
    error_log_path,
    maxBytes=10485760,  # 10 MB
    backupCount=5,  # Keep 5 backup logs
    encoding="utf-8",
)

# Set the file handler to only log errors and critical messages
file_handler.setLevel(logging.ERROR)

# Create a formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d"
)
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)


# Function to log messages based on verbosity setting
def log_message(message, is_step=False, is_summary=False, verbosity=True):
    """
    Log messages with verbosity control.

    Args:
        message: The message to log
        is_step: Whether this is a step message (always shown regardless of verbosity)
        is_summary: Whether this is a summary message (always shown regardless of verbosity)
        verbosity: Current verbosity setting from the API request
    """
    # Always log step and summary messages
    if is_step or is_summary or verbosity:
        logger.info(message)


# Cache for detected vLLM model name
_detected_vllm_model: Optional[str] = None


def _detect_vllm_model() -> str:
    """Detect the model served by the remote embedding endpoint.

    Calls /v1/models and returns the first model ID.
    Caches the result for subsequent calls.

    Returns:
        str: The model ID served by the endpoint.

    Raises:
        RuntimeError: If the endpoint is unreachable or returns no models.
    """
    global _detected_vllm_model
    if _detected_vllm_model:
        return _detected_vllm_model

    if not EMBEDDING_BASE_URL:
        raise RuntimeError("EMBEDDING_BASE_URL not configured")

    url = f"{EMBEDDING_BASE_URL}/models"
    headers = {"Content-Type": "application/json"}
    if EMBEDDING_API_KEY:
        headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            if models:
                _detected_vllm_model = models[0]["id"]
                return _detected_vllm_model
            raise RuntimeError("No models returned from /v1/models")
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Model detection failed: HTTP {exc.response.status_code}")
    except httpx.ConnectError as exc:
        raise RuntimeError(f"Model detection failed: {exc}")


def _vllm_request_with_retry(
    url: str,
    payload: dict,
    headers: dict,
    max_retries: int = EMBEDDING_MAX_RETRIES,
    base_delay: float = EMBEDDING_RETRY_BASE_DELAY,
) -> dict:
    """Make an HTTP POST with retry logic (429 exponential backoff, 5xx flat retry).

    Args:
        url: Full endpoint URL.
        payload: JSON request body.
        headers: HTTP headers.
        max_retries: Maximum retry count.
        base_delay: Base delay for backoff.

    Returns:
        dict: Parsed JSON response.

    Raises:
        HTTPException: On non-transient errors or retries exhausted.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=float(EMBEDDING_TIMEOUT)) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                wait = (2 ** attempt) * base_delay
                logger.warning(f"Rate limited (429), retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                last_error = exc
            elif status >= 500:
                logger.warning(f"Server error ({status}), retrying in {base_delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(base_delay)
                last_error = exc
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Embedding API error: HTTP {status}",
                )
        except httpx.ConnectError as exc:
            logger.warning(f"Connection error, retrying in {base_delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(base_delay)
            last_error = exc

    raise HTTPException(
        status_code=502,
        detail=f"Embedding API unavailable after {max_retries} retries: {last_error}",
    )


def get_embeddings_vllm(
    doc: List[str],
    model: str = None,
    **kwargs,
) -> dict:
    """Get embeddings via a remote OpenAI-compatible embedding API.

    Sends a single batch request for all input texts. Supports retry
    with exponential backoff on 429 and flat retry on 5xx.

    Args:
        doc: List of text strings to embed.
        model: Model name override. Auto-detected if not provided.

    Returns:
        dict: Mapping of input text to normalised embedding vector.

    Raises:
        HTTPException: On API errors after retries exhausted.
    """
    if not doc:
        return {}

    if not EMBEDDING_BASE_URL:
        raise HTTPException(
            status_code=500,
            detail="EMBEDDING_BASE_URL not configured",
        )

    # Resolve model name — "auto" or empty triggers detection from vLLM
    model_name = model
    if not model_name or model_name == "auto":
        model_name = EMBEDDING_MODEL_NAME or _detect_vllm_model()

    url = f"{EMBEDDING_BASE_URL}/embeddings"
    headers = {"Content-Type": "application/json"}
    if EMBEDDING_API_KEY:
        headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"

    payload = {
        "input": doc,
        "model": model_name,
    }

    data = _vllm_request_with_retry(url, payload, headers)

    # Build result dict, normalise vectors, re-order by index
    embeddings_dict = {}
    for item in data.get("data", []):
        idx = item.get("index", 0)
        if 0 <= idx < len(doc):
            vec = np.array(item["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings_dict[doc[idx]] = vec.tolist()

    return embeddings_dict


def split_into_sentences(doc: str) -> List[str]:
    """Split a document into sentences using regex pattern matching.

    Args:
        doc (str): The input document text to be split into sentences.

    Returns:
        List[str]: A list of sentences, with each sentence stripped of leading/trailing whitespace.

    Note:
        The function handles common edge cases like:
        - Titles (Mr., Mrs., Dr., etc.)
        - Common abbreviations (i.e., e.g., etc.)
        - Decimal numbers
        - Ellipsis
        - Quotes and brackets
    """
    # Define a pattern that looks for sentence boundaries but doesn't include them in the split
    # Instead of splitting directly at punctuation, we'll look for patterns that indicate sentence endings
    pattern = r"""
        # Match sentence ending punctuation followed by space and capital letter
        # Negative lookbehind for common titles and abbreviations
        (?<![A-Z][a-z]\.)                                                  # Not an abbreviation like U.S.
        (?<!Mr\.)(?<!Mrs\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ms\.) # Not a title
        (?<!i\.e\.)(?<!e\.g\.)(?<!vs\.)(?<!etc\.)                          # Not a common abbreviation
        (?<!\d\.)(?<!\.\d)                                                 # Not a decimal or numbered list
        (?<!\.\.\.)                                                        # Not an ellipsis
        [\.!\?]                                                            # Sentence ending punctuation
        \s+                                                                # One or more whitespace
        (?=[A-Z])                                                          # Followed by capital letter
    """

    # Find all positions where we should split
    split_positions = []
    for match in re.finditer(pattern, doc, re.VERBOSE):
        # Split after the punctuation and space
        split_positions.append(match.end())

    # Use the positions to extract sentences
    sentences = []
    start = 0
    for pos in split_positions:
        if pos > start:
            sentences.append(doc[start:pos].strip())
            start = pos

    # Add the last sentence if there's remaining text
    if start < len(doc):
        sentences.append(doc[start:].strip())

    # Filter out empty sentences
    return [s for s in sentences if s]


def get_embeddings(
    doc: List[str],
    model: str = EMBEDDER_MODEL,
    **kwargs,
) -> dict:
    """Generate embeddings using the remote vLLM API.

    Args:
        doc: List of text strings to generate embeddings for.
        model: Model name override.

    Returns:
        dict: Dictionary mapping input strings to their embeddings.
    """
    return get_embeddings_vllm(doc, model=model)


def calculate_similarity(
    embeddings_dict: dict, sentences: List[str]
) -> List[float]:
    """Calculate angular distance between consecutive sentence embeddings.

    Uses numpy for all computation — no torch/GPU required.

    Args:
        embeddings_dict: Dictionary mapping sentences to embedding vectors.
        sentences: Sentences in original order.

    Returns:
        List of angular distances between consecutive vector pairs.
    """
    if len(sentences) <= 1:
        return []

    try:
        vectors = np.array(
            [embeddings_dict[sentence] for sentence in sentences],
            dtype=np.float32,
        )

        # Consecutive pairs
        v1 = vectors[:-1]
        v2 = vectors[1:]

        # Norms
        norms1 = np.linalg.norm(v1, axis=1)
        norms2 = np.linalg.norm(v2, axis=1)

        # Cosine similarity → angular distance
        dot_products = np.sum(v1 * v2, axis=1)
        # Avoid division by zero
        denom = norms1 * norms2
        denom = np.where(denom == 0, 1.0, denom)
        cosine_sim = dot_products / denom
        angular_distance = 1 - cosine_sim

        return [round(float(d), 5) for d in angular_distance]

    except Exception as e:
        logger.error(f"Error calculating similarities: {str(e)}")
        raise HTTPException(status_code=500, detail="Error calculating similarities")


def _count_tokens_for_text(args: tuple[str, str]) -> int:
    """Count tokens in a text string using the specified encoding.

    Args:
        args (tuple): Tuple containing:
            - text (str): Text to count tokens for
            - encoding_name (str): Name of the tiktoken encoding to use

    Returns:
        int: Number of tokens in the text
    """
    text, encoding_name = args
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        # Fallback to approximate count if tiktoken fails
        return len(text.split())


def _group_chunks_by_similarity(
    sentences: List[str], distance: List[float], percentile: int
) -> tuple[dict[str, int], int, float, float]:
    """Group sentences into chunks based on similarity distances and a percentile threshold.

    Args:
        sentences (List[str]): List of sentences to group into chunks.
        distance (List[float]): List of similarity distances between consecutive sentences.
        percentile (int): Percentile value to use as threshold for chunk boundaries.

    Returns:
        tuple[dict[str, int], int, float, float]: A tuple containing:
            - Dictionary mapping text chunks to their token counts
            - Maximum token count across all chunks
            - Average token count across all chunks
            - Standard deviation of token counts
    """
    try:
        breakpoint = np.percentile(distance, percentile)
        indices_above_th = [i for i, x in enumerate(distance) if x > breakpoint]

        chunks = []
        start_index = 0
        for index in indices_above_th:
            combined_text = " ".join(sentences[start_index : index + 1])
            chunks.append(combined_text)
            start_index = index + 1

        if start_index < len(sentences):
            chunks.append(" ".join(sentences[start_index:]))

        # Calculate token counts with multiprocessing for large documents
        if len(chunks) > 100:
            with ProcessPoolExecutor(
                max_workers=min(MAX_WORKERS, len(chunks) // 10 + 1)
            ) as executor:
                token_args = [(chunk_text, "cl100k_base") for chunk_text in chunks]
                token_counts = list(executor.map(_count_tokens_for_text, token_args))
        else:
            # For smaller sets, avoid the overhead of multiprocessing
            token_counts = [
                _count_tokens_for_text((chunk_text, "cl100k_base"))
                for chunk_text in chunks
            ]

        # Create dictionary mapping chunks to token counts
        chunks_with_tokens = {
            chunk: count for chunk, count in zip(chunks, token_counts)
        }

        max_tokens = max(token_counts) if token_counts else 0
        average_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        std_dev = np.std(token_counts) if len(token_counts) > 1 else 0

        return chunks_with_tokens, max_tokens, average_tokens, std_dev

    except Exception as e:
        logger.error(f"Error in _group_chunks_by_similarity: {str(e)}")
        # Return safe fallback in case of error
        if not sentences:
            return {}, 0, 0, 0

        # Single chunk fallback
        combined_text = " ".join(sentences)
        token_count = _count_tokens_for_text((combined_text, "cl100k_base"))
        return {combined_text: token_count}, token_count, token_count, 0


def _process_percentile_range(
    args: tuple[List[str], List[float], int, int],
) -> tuple[Optional[dict[str, int]], Optional[int], Optional[float]]:
    """Process a single percentile and check if it produces valid chunks.

    Args:
        args (tuple): Tuple containing:
            - sentences (List[str]): List of sentences to process
            - distance (List[float]): List of similarity distances
            - max_tokens (int): Maximum tokens allowed per chunk
            - percentile (int): Percentile to check

    Returns:
        tuple[Optional[dict[str, int]], Optional[int], Optional[float]]:
            - Dictionary mapping chunks to token counts (if valid)
            - Percentile used (if valid)
            - Average tokens (if valid)
            - Or (None, None, None) if invalid
    """
    try:
        sentences, distance, max_tokens, percentile = args
        chunks_with_tokens, threshold_tokens, average_tokens, std_dev = (
            _group_chunks_by_similarity(sentences, distance, percentile)
        )

        # Calculate 95th percentile value using z-score of 1.645
        estimated_95th_percentile = average_tokens + (1.645 * std_dev)

        if estimated_95th_percentile <= max_tokens:
            return chunks_with_tokens, percentile, average_tokens

        return None, None, None

    except Exception as e:
        logger.error(f"Error processing percentile {percentile}: {str(e)}")
        return None, None, None


def _find_optimal_chunks(
    sentences: List[str], distance: List[float], max_tokens: int
) -> tuple[dict[str, int], int, float]:
    """Find optimal chunk groupings that fit within a token limit using statistical approach.

    Args:
        sentences (List[str]): List of sentences to group into chunks.
        distance (List[float]): List of similarity distances between consecutive sentences.
        max_tokens (int): Maximum number of tokens allowed per chunk.

    Returns:
        tuple[dict[str, int], int, float]: A tuple containing:
            - Dictionary mapping text chunks to their token counts
            - Percentile value used for grouping (0 if no suitable grouping found)
            - Average token count across all chunks

    Note:
        Uses a statistical approach by calculating the estimated 95th percentile
        of token counts to ensure most chunks stay below the token limit.
    """
    # Usa un intervallo di percentili più mirato per un'esplorazione più efficiente
    percentile_steps = 5
    for percentile in range(95, 0, -percentile_steps):
        chunks_with_tokens, max_token_val, average_tokens, std_dev = (
            _group_chunks_by_similarity(sentences, distance, percentile)
        )

        # Calculate 95th percentile value using z-score of 1.645
        estimated_95th_percentile = average_tokens + (1.645 * std_dev)

        if estimated_95th_percentile <= max_tokens:
            logger.info(
                f"Sequential fallback found valid percentile {percentile} with 95th percentile estimate {estimated_95th_percentile:.2f} <= {max_tokens}"
            )
            return chunks_with_tokens, percentile, average_tokens

    # If no valid percentile found, return the entire text as a single chunk
    logger.info(
        "No valid chunking found using sequential approach - returning single chunk"
    )
    fallback_chunks = {
        " ".join(sentences): _count_tokens_for_text(
            (" ".join(sentences), "cl100k_base")
        )
    }
    return fallback_chunks, 0, 0


def parallel_find_optimal_chunks(
    sentences: List[str],
    distance: List[float],
    max_tokens: int,
    start_percentile: int = 99,
    verbosity: bool = True,
) -> tuple[dict[str, int], int, float]:
    """Find optimal chunks using parallel batch processing starting from highest percentiles.

    Uses available CPU cores to process multiple percentiles simultaneously, collecting all
    valid results and selecting the highest percentile that satisfies the constraints.

    Args:
        sentences: List of sentences to group
        distance: List of similarity distances
        max_tokens: Maximum tokens per chunk
        start_percentile: Starting percentile for search (default: 99)

    Returns:
        tuple[dict[str, int], int, float]: Dictionary mapping chunks to token counts,
        percentile used, and average tokens
    """
    # Get available memory in GB and estimate memory needs based on document size
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    if verbosity:
        logger.info(f"Available system memory: {available_memory_gb:.2f} GB")

    # Estimate memory needed per worker (rough heuristic based on sentence count)
    estimated_mem_per_worker_gb = max(
        BASE_MEMORY_PER_WORKER_GB, len(sentences) * MEMORY_PER_SENTENCE_GB
    )
    if verbosity:
        logger.info(
            f"Estimated memory per worker: {estimated_mem_per_worker_gb:.2f} GB"
        )

    # Calculate max workers based on available memory (leave RESERVED_SYSTEM_MEMORY_GB for system)
    # Ensure estimated_mem_per_worker_gb is not zero to prevent DivisionByZeroError
    if estimated_mem_per_worker_gb <= 0:
        logger.warning(
            "Estimated memory per worker is zero or negative, defaulting memory_based_max_workers to 1."
        )
        memory_based_max_workers = 1
    else:
        memory_based_max_workers = max(
            1,
            int(
                (available_memory_gb - RESERVED_SYSTEM_MEMORY_GB)
                / estimated_mem_per_worker_gb
            ),
        )
    if verbosity:
        logger.info(f"Memory-based worker limit: {memory_based_max_workers}")

    # Also consider document size for worker scaling
    doc_size_factor = min(1.0, DOC_SIZE_FACTOR_SCALER / max(1, len(sentences)))
    cpu_based_max_workers = max(
        1, min(MAX_WORKERS, int(multiprocessing.cpu_count() * doc_size_factor))
    )
    if verbosity:
        logger.info(f"CPU-based worker limit: {cpu_based_max_workers}")

    # Take the minimum of memory-based and CPU-based worker counts
    max_workers = min(memory_based_max_workers, cpu_based_max_workers)

    # Hard cap based on sentence count for very large documents as extra safety
    if len(sentences) > VERY_LARGE_DOC_SENTENCE_THRESHOLD:
        max_workers = min(max_workers, WORKERS_VERY_LARGE_DOC)
        if verbosity:
            logger.info(
                f"Very large document (>{VERY_LARGE_DOC_SENTENCE_THRESHOLD} sentences), capping at {WORKERS_VERY_LARGE_DOC} worker(s)"
            )
    elif len(sentences) > LARGE_DOC_SENTENCE_THRESHOLD:
        max_workers = min(max_workers, WORKERS_LARGE_DOC)
        if verbosity:
            logger.info(
                f"Large document (>{LARGE_DOC_SENTENCE_THRESHOLD} sentences), capping at {WORKERS_LARGE_DOC} worker(s)"
            )

    # For very small documents, skip parallel processing
    if len(sentences) < MIN_SENTENCES_FOR_PARALLEL:
        if verbosity:
            logger.info(
                f"Small document (<{MIN_SENTENCES_FOR_PARALLEL} sentences), using sequential processing"
            )
        return _find_optimal_chunks(sentences, distance, max_tokens)

    if verbosity:
        logger.info(f"Using {max_workers} workers for parallel processing")

    try:
        # Scale step size inversely with document size
        if len(sentences) > STEP_SIZE_VERY_LARGE_DOC_THRESHOLD:
            initial_step_size = STEP_SIZE_VERY_LARGE_DOC
        elif len(sentences) > STEP_SIZE_LARGE_DOC_THRESHOLD:
            initial_step_size = STEP_SIZE_LARGE_DOC
        else:
            initial_step_size = STEP_SIZE_DEFAULT

        if verbosity:
            logger.info(f"Using initial step size of {initial_step_size}")

        for batch_start in range(start_percentile, 0, -initial_step_size * max_workers):
            batch_end = max(0, batch_start - initial_step_size * max_workers)
            current_percentiles = range(batch_start, batch_end, -initial_step_size)

            process_args = [
                (sentences, distance, max_tokens, p) for p in current_percentiles
            ]

            valid_results = []

            # Use a timeout for worker processes to prevent hung processes
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_process_percentile_range, args)
                    for args in process_args
                ]

                for future in as_completed(futures):
                    try:
                        chunks_with_tokens, percentile, average_tokens = future.result()
                        if chunks_with_tokens is not None:
                            valid_results.append(
                                (chunks_with_tokens, percentile, average_tokens)
                            )
                    except Exception as e:
                        logger.error(f"Error processing future: {str(e)}")
                        continue

            if valid_results:
                # We found at least one valid percentile, now refine with finer grain
                valid_results.sort(key=lambda x: x[1], reverse=True)
                best_valid_percentile = valid_results[0][1]

                # Perform a refined search around the best valid percentile
                refined_start = min(99, best_valid_percentile + initial_step_size)
                refined_end = max(1, best_valid_percentile - initial_step_size)
                refined_percentiles = range(refined_start, refined_end, -1)

                refined_args = [
                    (sentences, distance, max_tokens, p) for p in refined_percentiles
                ]
                refined_results = []

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(_process_percentile_range, args)
                        for args in refined_args
                    ]

                    for future in as_completed(futures):
                        try:
                            chunks_with_tokens, percentile, average_tokens = (
                                future.result()
                            )
                            if chunks_with_tokens is not None:
                                refined_results.append(
                                    (chunks_with_tokens, percentile, average_tokens)
                                )
                        except Exception as e:
                            logger.error(
                                f"Error processing future in refinement: {str(e)}"
                            )
                            continue

                # Combine results and select the best one
                all_results = valid_results + refined_results
                all_results.sort(key=lambda x: x[1], reverse=True)
                best_chunks_with_tokens, best_percentile, best_average_tokens = (
                    all_results[0]
                )

                if verbosity:
                    logger.info(
                        f"Selected the highest valid percentile: {best_percentile}"
                    )
                return best_chunks_with_tokens, best_percentile, best_average_tokens

        if verbosity:
            logger.info(
                "No valid chunking found in parallel processing, falling back to sequential"
            )
        return _find_optimal_chunks(sentences, distance, max_tokens)

    except Exception as e:
        logger.error(
            f"Error in parallel processing: {str(e)}, falling back to sequential approach"
        )
        return _find_optimal_chunks(sentences, distance, max_tokens)


def merge_undersized_chunks(
    chunks: List[dict],
    min_token_threshold: float,
    max_tokens: int,
    model: str = EMBEDDER_MODEL,
    verbosity: bool = False,
) -> List[dict]:
    """
    Merge chunks that are below a minimum token threshold with semantically similar neighbors.

    Args:
        chunks (List[dict]): List of chunks with 'text' and 'token_count' keys
        min_token_threshold (float): Minimum token threshold (e.g., 5th percentile)
        max_tokens (int): Maximum allowed tokens for a chunk
        model (str): Embedding model to use
        verbosity (bool): If True, shows all log messages and progress bars

    Returns:
        List[dict]: Updated list of chunks after merging small ones
    """
    # Step 1: Identify undersized chunks
    undersized_indices = [
        i
        for i, chunk in enumerate(chunks)
        if chunk["token_count"] < min_token_threshold
    ]

    if not undersized_indices:
        return chunks  # No small chunks to merge

    total_undersized = len(undersized_indices)

    # If most chunks are undersized, adjust the threshold
    if total_undersized > len(chunks) * 0.5:
        logger.info("Too many undersized chunks, adjusting threshold")
        min_token_threshold = min_token_threshold * 0.8
        undersized_indices = [
            i
            for i, chunk in enumerate(chunks)
            if chunk["token_count"] < min_token_threshold
        ]
        total_undersized = len(undersized_indices)

    # Step 2: Sort undersized chunks by token count (ascending) to process smallest first
    undersized_indices.sort(key=lambda i: chunks[i]["token_count"])

    # Step 3: Calculate initial embeddings for all chunks
    # We process all chunks in a single batch for efficiency
    chunk_texts = [chunk["text"] for chunk in chunks]

    # Get embeddings for all chunks in parallel using batch processing
    embeddings_dict = get_embeddings(
        doc=chunk_texts,
        model=model,
        batch_size=8,  # Process 8 chunks at a time for optimal performance
        verbosity=verbosity,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalized for cosine similarity
    )

    # Convert to numpy array while preserving the original chunk order
    embeddings = np.array([embeddings_dict[text] for text in chunk_texts])

    # Step 4: Initialize data structures for processing
    result_chunks = list(chunks)  # Working copy of chunks that will be modified
    merged_indices = set()  # Tracks indices of chunks that have been merged into others

    # Step 5: Process undersized chunks to find best merge candidates
    # We'll track which chunks need their embeddings updated after merging
    indices_to_update = []  # Stores indices of chunks that were merged
    texts_to_update = []  # Stores new text content for merged chunks

    for idx in undersized_indices:
        if idx in merged_indices:
            continue  # Skip if this chunk has already been merged

        current_chunk = result_chunks[idx]
        candidates = []

        # Check previous chunk if available
        if idx > 0 and idx - 1 not in merged_indices:
            prev_chunk = result_chunks[idx - 1]
            combined_tokens = current_chunk["token_count"] + prev_chunk["token_count"]

            if combined_tokens <= max_tokens:
                # Calculate cosine similarity using dot product (vectors are already normalized)
                similarity = float(np.dot(embeddings[idx], embeddings[idx - 1]))
                candidates.append((idx - 1, similarity, combined_tokens))

        # Check next chunk if available
        if idx < len(result_chunks) - 1 and idx + 1 not in merged_indices:
            next_chunk = result_chunks[idx + 1]
            combined_tokens = current_chunk["token_count"] + next_chunk["token_count"]

            if combined_tokens <= max_tokens:
                # Calculate cosine similarity
                similarity = float(np.dot(embeddings[idx], embeddings[idx + 1]))
                candidates.append((idx + 1, similarity, combined_tokens))

        # If no valid candidates, continue to next chunk
        if not candidates:
            continue

        # Find best candidate based on similarity
        best_candidate = max(candidates, key=lambda x: x[1])
        merge_idx, similarity, combined_tokens = best_candidate

        # Determine merge order (maintain document order)
        if merge_idx < idx:
            merged_text = result_chunks[merge_idx]["text"] + " " + current_chunk["text"]
            target_idx = merge_idx
            removed_idx = idx
        else:
            merged_text = current_chunk["text"] + " " + result_chunks[merge_idx]["text"]
            target_idx = idx
            removed_idx = merge_idx

        # Update chunk at target index
        result_chunks[target_idx] = {
            "text": merged_text,
            "token_count": combined_tokens,
        }

        # Mark the other chunk as None (to be filtered later)
        result_chunks[removed_idx] = None

        # Mark the removed chunk as merged to prevent future merges with it
        merged_indices.add(removed_idx)

        # Track this merge for batch embedding update
        indices_to_update.append(target_idx)  # The index that was updated
        texts_to_update.append(merged_text)  # The new combined text

    # Step 6: Update embeddings for all merged chunks in a single batch
    # This is more efficient than updating embeddings one at a time
    if texts_to_update:
        # Get new embeddings for all merged chunks in parallel
        new_embeddings_dict = get_embeddings(
            doc=texts_to_update,
            model=model,
            batch_size=8,  # Process up to 8 chunks at once
            verbosity=verbosity,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Maintains cosine similarity compatibility
        )

        # Update the embeddings array with new values for merged chunks
        for i, text in zip(indices_to_update, texts_to_update):
            embeddings[i] = new_embeddings_dict[text]

    # Filter out None values (merged chunks)
    final_chunks = [chunk for chunk in result_chunks if chunk is not None]

    # Report merging statistics
    if verbosity:
        logger.info(
            f"After merging: {len(chunks)} → {len(final_chunks)} chunks ({len(chunks) - len(final_chunks)} merged), {total_undersized} originally undersized"
        )

    return final_chunks


def _split_large_sentence(
    sentence: str, max_tokens: int
) -> List[Dict[str, Union[str, int]]]:
    """Split an oversized sentence at token boundaries to fit within token limits.

    Args:
        sentence: The sentence text to split
        max_tokens: Maximum number of tokens per chunk

    Returns:
        List of dictionaries containing text and token_count
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        all_tokens = encoding.encode(sentence)

        chunks = []
        for i in range(0, len(all_tokens), max_tokens):
            chunk_tokens = all_tokens[i : min(i + max_tokens, len(all_tokens))]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append({"text": chunk_text, "token_count": len(chunk_tokens)})

        return chunks
    except Exception as e:
        logger.error(f"Error splitting large sentence: {str(e)}")
        # Fallback to a simple character-based split
        total_chars = len(sentence)
        avg_chars_per_token = 4  # Rough estimate
        chars_per_chunk = max_tokens * avg_chars_per_token

        chunks = []
        for i in range(0, total_chars, chars_per_chunk):
            chunk_text = sentence[i : min(i + chars_per_chunk, total_chars)]
            # Estimate token count
            token_count = len(chunk_text) // avg_chars_per_token
            chunks.append({"text": chunk_text, "token_count": token_count})

        return chunks


def split_oversized_chunk(
    chunk_text: str, max_tokens: int
) -> List[Dict[str, Union[str, int]]]:
    """Split an oversized chunk using a sentence-based approach with semantic prioritization.

    Args:
        chunk_text: The text to be split into chunks
        max_tokens: Maximum number of tokens allowed per chunk

    Returns:
        List of dictionaries containing text and token_count
    """
    try:
        # Split text into sentences
        sentences = split_into_sentences(chunk_text)

        # Calculate token count for each sentence
        sentence_tokens = [
            (s, _count_tokens_for_text((s, "cl100k_base"))) for s in sentences
        ]

        # Create chunks respecting sentence boundaries
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Optimal target is 70-80% of max_tokens for better balance
        target_size = int(max_tokens * 0.75)

        for sentence, tokens in sentence_tokens:
            # If the sentence is already too large (rare), split at token level
            if tokens > max_tokens:
                chunks.extend(_split_large_sentence(sentence, max_tokens))
                continue

            # If adding this sentence would exceed max_tokens, finalize current chunk
            if current_tokens + tokens > max_tokens:
                chunks.append(
                    {"text": " ".join(current_chunk), "token_count": current_tokens}
                )
                current_chunk = []
                current_tokens = 0

            # If we've reached optimal size and are at a "natural" sentence boundary
            elif (
                current_tokens >= target_size
                and sentence[-1] in ".!?"
                and len(current_chunk) > 0
            ):
                chunks.append(
                    {"text": " ".join(current_chunk), "token_count": current_tokens}
                )
                current_chunk = []
                current_tokens = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += tokens

        # Add final chunk if there's anything remaining
        if current_chunk:
            chunks.append(
                {"text": " ".join(current_chunk), "token_count": current_tokens}
            )

        return chunks
    except Exception as e:
        logger.error(f"Error splitting oversized chunk: {str(e)}")
        # Fallback to simpler splitting method
        return _split_large_sentence(chunk_text, max_tokens)


def validate_file_extension(filename: str) -> None:
    """Validate that the file has an allowed extension.

    Args:
        filename: Name of the file to validate

    Raises:
        HTTPException: If the file extension is not allowed
    """
    if not filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    extension = filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '.{extension}' is not allowed. Only {', '.join(ALLOWED_EXTENSIONS)} files are accepted",
        )


def parse_json_content(content: bytes) -> List[str]:
    """Parse JSON content and extract text chunks.

    Args:
        content: Raw bytes from the uploaded file

    Returns:
        List of text strings from the JSON chunks

    Raises:
        ValueError: With specific messages for different validation failures
    """
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")

        if "chunks" not in data:
            raise ValueError("JSON must contain a 'chunks' key")

        if not isinstance(data["chunks"], list):
            raise ValueError("JSON 'chunks' must be an array")

        # Protect against excessive chunks
        MAX_CHUNKS = 10000  # Set a reasonable limit
        if len(data["chunks"]) > MAX_CHUNKS:
            raise ValueError(f"JSON contains too many chunks (max: {MAX_CHUNKS})")

        chunks = []
        for i, item in enumerate(data["chunks"], 1):
            if not isinstance(item, dict):
                raise ValueError(f"Chunk {i} must be an object")

            if "text" not in item:
                raise ValueError(f"Chunk {i} is missing required 'text' field")

            if not isinstance(item["text"], str):
                raise ValueError(f"Chunk {i} 'text' must be a string")

            # Skip empty strings after stripping
            text = item["text"].strip()
            if text:
                chunks.append(text)

        if not chunks:
            raise ValueError("No valid text chunks found in JSON")

        # Il log dettagliato verrà gestito dalla funzione chiamante in base all'impostazione di verbosità
        return chunks

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {str(e)}")
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Error parsing JSON: {str(e)}")
        raise ValueError(f"Error parsing JSON: {str(e)}")


def validate_file_size(content_length: int) -> None:
    """Validate that the file size is within allowed limits.

    Args:
        content_length: Length of the file content in bytes

    Raises:
        HTTPException: If the file size exceeds the maximum allowed
    """
    if content_length > MAX_FILE_SIZE:
        max_size_mb = MAX_FILE_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds the maximum allowed size of {max_size_mb:.1f} MB",
        )


class ChunkingInput(BaseModel):
    max_tokens: int = Field(..., gt=0, description="Maximum number of tokens per chunk")
    model: str = Field(
        default=EMBEDDER_MODEL,
        description="Embedding model to use. Default 'auto' detects from vLLM server.",
        json_schema_extra={"example": "auto"},
    )
    merge_small_chunks: bool = Field(
        default=True,
        description="Whether to merge undersized chunks for more balanced output",
    )
    verbosity: bool = Field(
        default=False,
        description="If True, shows all log messages. If False, only shows step headers and final statistics.",
    )


class ChunkingMetadata(BaseModel):
    n_chunks: int
    avg_tokens: float
    max_tokens: int
    min_tokens: int
    percentile: int
    embedder_model: str
    source: str
    processing_time: float


class Chunk(BaseModel):
    text: str
    token_count: int
    id: int


class ChunkingResult(BaseModel):
    chunks: List[Chunk]
    metadata: ChunkingMetadata


@app.get("/")
async def health_check():
    """Check the health status of the API service.

    Returns:
        dict: A dictionary containing:
            - status: Current health status of the service
            - embedding_backend: Which embedding backend is active
            - version: Current API version
    """
    result = {
        "status": "healthy",
        "embedding_backend": "vllm",
        "version": app.version,
    }

    try:
        model_name = _detect_vllm_model()
        result["embedding_model"] = model_name
        result["embedding_endpoint"] = "connected"
    except Exception:
        result["embedding_endpoint"] = "unreachable"

    return result


@app.post("/normalized_semantic_chunker/", response_model=ChunkingResult)
async def Normalized_Semantic_Chunker(
    file: UploadFile = File(...),
    input_data: ChunkingInput = Depends(),
):
    """
    Process and chunk a text document into smaller, semantically coherent segments using advanced NLP techniques.
    The function performs the following steps:
    1. Validates the input file format and size
    2. Processes the content based on file type
    3. Splits text into sentences using regex pattern matching
    4. Generates embeddings for semantic analysis
    5. Groups sentences into chunks based on semantic similarity
    6. Handles edge cases like oversized chunks and merges small chunks when needed

    Args:
        file (UploadFile): The uploaded file to process. Supported formats:
            - Text files (.txt, .md): Plain text content that will be split into sentences
            - JSON files (.json): Expected format: {"chunks": [{"text": "..."}, ...]}
        input_data (ChunkingInput): Configuration parameters for the chunking process:
            - max_tokens (int): Maximum number of tokens allowed per chunk. Must be > 0.
            - model (str, optional): Name of the embedding model. Defaults to auto-detected model from the vLLM server.
            - merge_small_chunks (bool, optional): If True, merges undersized chunks with semantically similar neighbors. Defaults to True.
            - show_progress_bar (bool, optional): If True, displays progress bars during long-running operations. Defaults to False.

    Returns:
        ChunkingResult: Object containing the chunking results with the following structure:
            - chunks (List[Chunk]): List of processed chunks, each containing:
                - text (str): The chunk content
                - token_count (int): Number of tokens in the chunk
                - id (int): Unique identifier for the chunk
            - metadata (ChunkingMetadata): Processing statistics and metadata:
                - n_chunks (int): Total number of chunks created
                - avg_tokens (float): Average token count across all chunks
                - max_tokens (int): Maximum token count in any chunk
                - min_tokens (int): Minimum token count in any chunk
                - percentile (int): Optimal percentile used for semantic chunking
                - embedder_model (str): Name of the model used for embeddings
                - processing_time (float): Total processing time in seconds

    Raises:
        HTTPException: With appropriate status codes for various error conditions:
            - 400: Invalid file format, empty content, or invalid parameters
            - 413: File size exceeds maximum allowed limit
            - 500: Internal server error during processing
            - 422: Validation error for input parameters
    """
    start_time = time.time()

    try:
        # Validate file extension
        validate_file_extension(file.filename)

        # Read and validate file size
        content = await file.read()
        validate_file_size(len(content))

        # Check if file is JSON
        if file.filename.lower().endswith(".json"):
            try:
                # Parse JSON content directly
                logger.info("Step 1 - JSON file processing")
                sentences = parse_json_content(content)
                if input_data.verbosity:
                    logger.info(f"Extracted {len(sentences)} text chunks from JSON")
            except ValueError as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid JSON format: {str(e)}"
                )
        else:
            # Handle text files (txt, md)
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                text = content.decode("latin-1")

            # Step 1: Split the document into sentences
            logger.info("Step 1 - Text Splitting")
            try:
                if input_data.verbosity:
                    logger.info("Using regex-based text splitting")
                sentences = split_into_sentences(text)
                if input_data.verbosity:
                    logger.info(f"Number of sentences: {len(sentences)}")
            except ValueError as ve:
                # Handle validation errors (like None inputs or wrong types)
                logger.error(f"Validation error during sentence splitting: {str(ve)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid input for sentence splitting: {str(ve)}",
                )
            except RuntimeError as re:
                # Handle runtime errors from the sentence splitting functions
                logger.error(f"Runtime error during sentence splitting: {str(re)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing document with sentence splitting: {str(re)}",
                )
            except Exception as e:
                # Catch any other unexpected errors
                logger.error(
                    f"Unexpected error during sentence splitting (method: {'newline' if input_data.use_newline_splitting else 'regex'}): {str(e)}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected error during sentence splitting: {str(e)}",
                )

        if not sentences:
            raise HTTPException(
                status_code=400,
                detail="No valid sentences found in the document. Please check the content.",
            )

        # Step 2: Vector embedding
        logger.info("Step 2 - Vector Embedding")
        embeddings_dict = get_embeddings(
            sentences,
            model=input_data.model,
            verbosity=input_data.verbosity,
        )

        # Step 3: Calculate sentence vector distances
        logger.info("Step 3 - Sentence Vector Distances Calculation")
        sentence_vector_distance = calculate_similarity(embeddings_dict, sentences)

        # Step 4: Semantic chunking - now returns dictionary with token counts
        logger.info("Step 4 - Semantic Chunking")
        chunks_with_tokens, percentile, average_tokens = parallel_find_optimal_chunks(
            sentences,
            sentence_vector_distance,
            input_data.max_tokens,
            verbosity=input_data.verbosity,
        )

        # Check if valid percentile was found, otherwise raise exception
        if not percentile:
            logger.warning("No valid percentile found for chunking")
            raise HTTPException(
                status_code=400,
                detail="No valid percentile found for chunking. Try increasing max_tokens parameter.",
            )

        # Dict text->token_count
        final_chunks = []

        for chunk_text, token_count in chunks_with_tokens.items():
            final_chunks.append({"text": chunk_text, "token_count": token_count})

        # Extract token counts for statistical analysis
        token_counts = [chunk["token_count"] for chunk in final_chunks]
        total_chunks = len(token_counts)

        if not token_counts:
            raise HTTPException(
                status_code=500,
                detail="Chunking failed to produce any valid chunks. Please try with different parameters.",
            )

        # Calculate statistics
        mean_tokens = np.mean(token_counts)
        percentile_5th = np.percentile(token_counts, 5)
        percentile_95th = np.percentile(token_counts, 95)

        # Count chunks in each category
        chunks_under_5th = sum(1 for count in token_counts if count < percentile_5th)
        chunks_over_95th = sum(1 for count in token_counts if count > percentile_95th)
        chunks_in_range = total_chunks - chunks_under_5th - chunks_over_95th

        # STEP 5: Add to log
        logger.info("Step 5 - Distribution Statistics Calculation")

        # Statistiche dettagliate mostrate solo in modalità verbosa
        if input_data.verbosity:
            logger.info(f"Mean tokens: {mean_tokens:.2f}")
            logger.info(f"Total chunks: {total_chunks}")
            logger.info(f"5th percentile threshold: {percentile_5th:.2f} tokens")
            logger.info(f"95th percentile threshold: {percentile_95th:.2f} tokens")
            logger.info(
                f"Number of chunks under 5th percentile: {chunks_under_5th} ({chunks_under_5th / total_chunks * 100:.2f}%)"
            )
            logger.info(
                f"Number of chunks in 5-95th percentile: {chunks_in_range} ({chunks_in_range / total_chunks * 100:.2f}%)"
            )
            logger.info(
                f"Number of chunks over 95th percentile: {chunks_over_95th} ({chunks_over_95th / total_chunks * 100:.2f}%)"
            )

        # STEP 6: Merge undersized chunks if enabled
        if input_data.merge_small_chunks and chunks_under_5th > 0:
            logger.info("Step 6 - Merge Undersized Chunks")
            if input_data.verbosity:
                logger.info("Merge process for chunks under 5th percentile")
            final_chunks = merge_undersized_chunks(
                chunks=final_chunks,
                min_token_threshold=percentile_5th,
                max_tokens=input_data.max_tokens // 2,
                model=input_data.model,
                verbosity=input_data.verbosity,
            )

        # STEP 7: Split oversized chunks
        logger.info("Step 7 - Split Oversized Chunks")
        oversized_indices = [
            i
            for i, chunk in enumerate(final_chunks)
            if chunk["token_count"] > input_data.max_tokens
        ]

        if oversized_indices:
            if input_data.verbosity:
                logger.info("Starting split process for oversized chunks")
                logger.info(
                    f"Found {len(oversized_indices)} chunks over threshold of {input_data.max_tokens} tokens"
                )

        normalized_chunks = []

        for i, chunk in enumerate(final_chunks):
            if i in oversized_indices:
                sub_chunks = split_oversized_chunk(chunk["text"], input_data.max_tokens)
                normalized_chunks.extend(sub_chunks)
            else:
                normalized_chunks.append(chunk)

        chunks_before_split = len(final_chunks)
        final_chunks = normalized_chunks

        token_counts_after_split = [chunk["token_count"] for chunk in final_chunks]

        if not token_counts_after_split:
            raise HTTPException(
                status_code=500,
                detail="Chunking failed after splitting process. Please try with different parameters.",
            )

        mean_tokens_after_split = np.mean(token_counts_after_split)
        max_tokens_after_split = max(token_counts_after_split)

        if input_data.verbosity:
            logger.info(
                f"After splitting: {len(final_chunks)} total chunks (increased from {chunks_before_split} to {len(final_chunks)}, +{len(final_chunks) - chunks_before_split})"
            )

        logger.info("Step 8 - Semantic Chunking Statistics after processing")

        # Statistiche finali mostrate sempre, indipendentemente dalla verbosity
        logger.info(f"Mean tokens: {mean_tokens_after_split:.2f}")
        logger.info(f"Total chunks: {len(final_chunks)}")
        logger.info(f"Smallest chunk: {min(token_counts_after_split)} tokens")
        logger.info(f"Largest chunk: {max_tokens_after_split} tokens")

        processing_time = time.time() - start_time
        result = ChunkingResult(
            chunks=[
                Chunk(text=chunk["text"], token_count=chunk["token_count"], id=i + 1)
                for i, chunk in enumerate(final_chunks)
            ],
            metadata=ChunkingMetadata(
                n_chunks=len(final_chunks),
                avg_tokens=float(
                    sum(chunk["token_count"] for chunk in final_chunks)
                    / len(final_chunks)
                ),
                max_tokens=max(chunk["token_count"] for chunk in final_chunks),
                min_tokens=min(chunk["token_count"] for chunk in final_chunks),
                percentile=percentile,
                embedder_model=input_data.model,
                source=file.filename,  # Use filename as the source
                processing_time=processing_time,
            ),
        )

        end_time = time.time()
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        return result

    except HTTPException:
        # Re-raise HTTP exceptions as is
        raise
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f"Error processing request: {error_type} - {error_msg}")

        # Return a safe error message without exposing system details
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the chunking request: {error_type}. Please check the logs for more information.",
        )
