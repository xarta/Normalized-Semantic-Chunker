import time
import logging
import numpy as np
import re
import tiktoken
import torch
import multiprocessing
import os
from logging.handlers import RotatingFileHandler
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from contextlib import asynccontextmanager

# Configurazione modificabile tramite variabili d'ambiente
ALLOWED_EXTENSIONS = {"txt", "md"}
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB di default
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", min(multiprocessing.cpu_count() - 1, 4)))
CACHE_TIMEOUT = int(os.environ.get("CACHE_TIMEOUT", 3600))  # 1 ora in secondi


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading embedding model {EMBEDDER_MODEL} during application startup...")
    try:
        _get_model(EMBEDDER_MODEL)
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
    
    yield
    
    # Cleanup
    logger.info("Application shutting down, cleaning up resources...")
    try:
        # Cleanup model cache
        with _model_lock:
            for model_name in list(_model_cache.keys()):
                if model_name in _model_cache:
                    del _model_cache[model_name]
                    logger.info(f"Removed model {model_name} from cache")
        
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared during shutdown")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


app = FastAPI(
    title="Normalized Semantic Chunker",
    description="API for processing and chunking text documents into smaller, semantically coherent segments",
    version="0.6.0",
    lifespan=lifespan,
)

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging
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

# Create a singleton for model caching with expiration
_model_cache = {}
_model_last_used = {}
_model_lock = multiprocessing.RLock()  # Thread-safe lock for model cache


def _get_model(model_name: str) -> SentenceTransformer:
    """Get model from cache or load it into RAM with cache expiration.

    Args:
        model_name (str): Name or path of the model to use.

    Returns:
        SentenceTransformer: The loaded model instance.
    """
    current_time = time.time()
    
    with _model_lock:
        # Check for expired models first
        expired_models = [
            name for name, last_used in _model_last_used.items() 
            if current_time - last_used > CACHE_TIMEOUT
        ]
        
        # Remove expired models
        for name in expired_models:
            if name in _model_cache and name != model_name:  # Don't remove the one we're about to use
                logger.info(f"Removing expired model {name} from cache")
                del _model_cache[name]
                del _model_last_used[name]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Update or load the requested model
        if model_name not in _model_cache:
            # Create models directory if it doesn't exist
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            # Local path for the model
            local_model_path = models_dir / model_name.replace("/", "_")

            try:
                if local_model_path.exists():
                    # Load from local storage
                    logger.info(f"Loading model from local storage: {local_model_path}")
                    _model_cache[model_name] = SentenceTransformer(str(local_model_path))
                else:
                    # Download and save model
                    logger.info(
                        f"Downloading model {model_name} and saving to {local_model_path}"
                    )
                    _model_cache[model_name] = SentenceTransformer(model_name)
                    _model_cache[model_name].save(str(local_model_path))
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                raise
        
        # Update last used timestamp
        _model_last_used[model_name] = current_time
        
        return _model_cache[model_name]


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
    batch_size: int = 8,  # Aumentato da 4 a 8 per migliorare la performance
    show_progress_bar: bool = False,  # Cambiato a False di default
    convert_to_numpy: bool = True,
    normalize_embeddings: bool = True,
) -> dict[str, List[float]]:
    """Generate embeddings for a list of text strings using a Sentence Transformer model.

    Args:
        doc (List[str]): List of text strings to generate embeddings for.
        model (str, optional): Name or path of the model to use.
        batch_size (int, optional): Batch size for embedding generation.
        show_progress_bar (bool, optional): Whether to show progress bar.
        convert_to_numpy (bool, optional): Whether to convert output to numpy array.
        normalize_embeddings (bool, optional): Whether to normalize embeddings.

    Returns:
        dict[str, List[float]]: Dictionary mapping input strings to their embeddings.

    Raises:
        HTTPException: If there's an error during the embedding process.
    """
    try:
        # Get model from cache (loads from disk if not in RAM)
        model_instance = _get_model(model)
        
        # Choose device and batch size appropriately
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Adjust batch size based on document size to prevent OOM errors
        if len(doc) > 1000:
            batch_size = max(1, batch_size // 2)
            logger.info(f"Large document detected, reducing batch size to {batch_size}")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model_instance = model_instance.to(device)

        # Get embeddings
        embeddings = model_instance.encode(
            doc,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )

        # Create dictionary mapping sentences to embeddings
        result = {
            sentence: embedding.tolist() for sentence, embedding in zip(doc, embeddings)
        }

        # Cleanup GPU memory but keep model in RAM
        if torch.cuda.is_available():
            model_instance.cpu()
            del embeddings
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        # Cleanup in case of error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embeddings: {str(e)}",
        )


def calculate_similarity(
    embeddings_dict: dict[str, List[float]], sentences: List[str]
) -> List[float]:
    """Calculate similarity scores between consecutive vectors in sentence order.

    Args:
        embeddings_dict (dict[str, List[float]]): Dictionary mapping sentences to vectors
        sentences (List[str]): Sentences in original order

    Returns:
        List[float]: List of similarity scores between consecutive vector pairs
    """
    if len(sentences) <= 1:
        return []

    try:
        # Extract vectors in sentence order
        vectors = [embeddings_dict[sentence] for sentence in sentences]

        # Process in batches for large documents to prevent OOM errors
        batch_size = 5000  # Adjust based on memory constraints
        if len(vectors) > batch_size:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            similarities = []
            
            for i in range(0, len(vectors) - 1, batch_size):
                end_idx = min(i + batch_size, len(vectors) - 1)
                batch_vectors1 = torch.tensor(vectors[i:end_idx], dtype=torch.float32, device=device)
                batch_vectors2 = torch.tensor(vectors[i+1:end_idx+1], dtype=torch.float32, device=device)
                
                # Calculate norms
                norms1 = torch.linalg.norm(batch_vectors1, dim=1)
                norms2 = torch.linalg.norm(batch_vectors2, dim=1)
                
                # Calculate dot products
                dot_products = torch.sum(batch_vectors1 * batch_vectors2, dim=1)
                
                # Calculate similarities
                batch_similarities = 1 - (dot_products / (norms1 * norms2))
                similarities.extend([round(float(sim), 5) for sim in batch_similarities.cpu()])
                
                # Clean up batch memory
                del batch_vectors1, batch_vectors2, norms1, norms2, dot_products, batch_similarities
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return similarities
        else:
            # For smaller documents, process all at once
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vectors_tensor = torch.tensor(vectors, dtype=torch.float32, device=device)
            
            # Get consecutive pairs
            vectors1 = vectors_tensor[:-1]
            vectors2 = vectors_tensor[1:]
            
            # Calculate norms
            norms = torch.linalg.norm(vectors_tensor, dim=1)
            norms1 = norms[:-1]
            norms2 = norms[1:]
            
            # Calculate dot products
            dot_products = torch.sum(vectors1 * vectors2, dim=1)
            
            # Calculate similarities (using 1 - cosine similarity for angular distance)
            cosine_similarities = dot_products / (norms1 * norms2)
            angular_similarities = 1 - cosine_similarities
            
            # Move to CPU and round
            result = [round(float(sim), 5) for sim in angular_similarities.cpu()]
            
            # Cleanup GPU memory
            if torch.cuda.is_available():
                del vectors_tensor, vectors1, vectors2, norms, norms1, norms2
                del dot_products, cosine_similarities, angular_similarities
                torch.cuda.empty_cache()
            
            return result

    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
            with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks)//10 + 1)) as executor:
                token_args = [(chunk_text, "cl100k_base") for chunk_text in chunks]
                token_counts = list(executor.map(_count_tokens_for_text, token_args))
        else:
            # For smaller sets, avoid the overhead of multiprocessing
            token_counts = [_count_tokens_for_text((chunk_text, "cl100k_base")) for chunk_text in chunks]

        # Create dictionary mapping chunks to token counts
        chunks_with_tokens = {chunk: count for chunk, count in zip(chunks, token_counts)}

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
    # Optimize worker count based on document size and available cores
    workers_scale_factor = min(1.0, len(sentences) / 5000)  # Scale workers based on document size
    max_workers = max(1, min(MAX_WORKERS, int(multiprocessing.cpu_count() * workers_scale_factor)))
    
    # For very small documents, skip parallel processing
    if len(sentences) < 100:
        logger.info("Small document detected, using sequential processing")
        return _find_optimal_chunks(sentences, distance, max_tokens)
    
    logger.info(f"Using {max_workers} workers for parallel processing")
    
    try:
        # Use larger steps for the initial search
        initial_step_size = 10
        
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
                            chunks_with_tokens, percentile, average_tokens = future.result()
                            if chunks_with_tokens is not None:
                                refined_results.append(
                                    (chunks_with_tokens, percentile, average_tokens)
                                )
                        except Exception as e:
                            logger.error(f"Error processing future in refinement: {str(e)}")
                            continue
                
                # Combine results and select the best one
                all_results = valid_results + refined_results
                all_results.sort(key=lambda x: x[1], reverse=True)
                best_chunks_with_tokens, best_percentile, best_average_tokens = all_results[0]
                
                logger.info(f"Selected the highest valid percentile: {best_percentile}")
                return best_chunks_with_tokens, best_percentile, best_average_tokens
        
        logger.info("No valid chunking found in parallel processing, falling back to sequential")
        return _find_optimal_chunks(sentences, distance, max_tokens)
    
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}, falling back to sequential approach")
        return _find_optimal_chunks(sentences, distance, max_tokens)


def merge_undersized_chunks(
    chunks: List[dict],
    min_token_threshold: float,
    max_tokens: int,
    model: str = EMBEDDER_MODEL,
) -> List[dict]:
    """
    Merge chunks that are below a minimum token threshold with semantically similar neighbors.

    Args:
        chunks (List[dict]): List of chunks with 'text' and 'token_count' keys
        min_token_threshold (float): Minimum token threshold (e.g., 5th percentile)
        max_tokens (int): Maximum allowed tokens for a chunk
        model (str): Embedding model to use

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

    # Step 3: Calculate embeddings for all chunks at once for efficiency
    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings_dict = get_embeddings(
        doc=chunk_texts,
        model=model,
        batch_size=8,  # Increased batch size
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Convert dictionary result to numpy array in the same order as chunk_texts
    embeddings = np.array([embeddings_dict[text] for text in chunk_texts])

    # Step 4: Create working copies for processing
    result_chunks = list(chunks)
    merged_indices = set()  # Track indices that have been merged

    # Step 5: Process each undersized chunk with adaptive similarity threshold
    base_similarity_threshold = 0.5  # Start with moderate threshold
    
    for idx in undersized_indices:
        if idx in merged_indices:
            continue  # Skip if this chunk has already been merged

        current_chunk = result_chunks[idx]
        candidates = []

        # Calculate dynamic similarity threshold based on chunk size
        # Smaller chunks can use lower similarity thresholds for merging
        size_factor = min(1.0, current_chunk["token_count"] / min_token_threshold)
        adaptive_threshold = base_similarity_threshold * size_factor
        
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

        # Mark indices as merged
        merged_indices.add(removed_idx)

        # Update embedding for the merged chunk to enable further merges
        new_embedding_dict = get_embeddings(
            doc=[merged_text],
            model=model,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings[target_idx] = new_embedding_dict[merged_text]

    # Filter out None values (merged chunks)
    final_chunks = [chunk for chunk in result_chunks if chunk is not None]

    # Report merging statistics
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
            chunk_text = sentence[i:min(i+chars_per_chunk, total_chars)]
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
            chunks.append({"text": " ".join(current_chunk), "token_count": current_tokens})

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
        description=f"Embedding model to use. Default is {EMBEDDER_MODEL}",
        json_schema_extra={"example": EMBEDDER_MODEL},
    )
    merge_small_chunks: bool = Field(
        default=True,
        description="Whether to merge undersized chunks for more balanced output",
    )


class ChunkingMetadata(BaseModel):
    n_chunks: int
    avg_tokens: float
    max_tokens: int
    min_tokens: int
    percentile: int
    embedder_model: str
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
            - gpu_available: Boolean indicating if GPU is available
            - version: Current API version
    """
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "version": app.version,
        "default_model": EMBEDDER_MODEL,
    }


@app.post("/normalized_semantic_chunker/", response_model=ChunkingResult)
async def Normalized_Semantic_Chunker(
    file: UploadFile = File(...), 
    input_data: ChunkingInput = Depends(),
    background_tasks: BackgroundTasks = None,
):
    """
    Process and chunk a text document into smaller, semantically coherent segments.

    Args:
        file (UploadFile): The uploaded text file to process (.txt or .md format).
        input_data (ChunkingInput): Input parameters including:
            - max_tokens (int): Maximum token count per chunk
            - model (str): Embedding model name to use for semantic analysis
            - merge_small_chunks (bool): Whether to merge undersized chunks

    Returns:
        ChunkingResult: Object containing:
            - chunks (List[Chunk]): List of text chunks with their token counts
            - metadata (ChunkingMetadata): Processing statistics including:
                - n_chunks: Total number of chunks created
                - avg_tokens: Average token count per chunk
                - max_tokens: Maximum tokens in any chunk
                - min_tokens: Minimum tokens in any chunk
                - percentile: Optimal percentile used for chunking
                - embedder_model: Model used for embeddings
                - processing_time: Total processing time in seconds

    Raises:
        HTTPException: 400 if file format is invalid or processing fails
    """
    start_time = time.time()

    try:
        # Validate file extension
        validate_file_extension(file.filename)
        
        # Read and validate file size
        content = await file.read()
        validate_file_size(len(content))
        
        # Decode file content
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            text = content.decode("latin-1")

        # Step 1: Split the document into sentences
        logger.info("Step 1 - Sentence Splitting")
        sentences = split_into_sentences(text)
        logger.info(f"Number of sentences: {len(sentences)}")
        
        if not sentences:
            raise HTTPException(
                status_code=400,
                detail="No valid sentences found in the document. Please check the content.",
            )

        # Step 2: Vector embedding
        logger.info("Step 2 - Vector Embedding")
        embeddings_dict = get_embeddings(sentences, input_data.model)

        # Step 3: Calculate sentence vector distances
        logger.info("Step 3 - Sentence Vector Distances Calculation")
        sentence_vector_distance = calculate_similarity(embeddings_dict, sentences)

        # Step 4: Semantic chunking - now returns dictionary with token counts
        logger.info("Step 4 - Semantic Chunking")
        chunks_with_tokens, percentile, average_tokens = parallel_find_optimal_chunks(
            sentences, sentence_vector_distance, input_data.max_tokens
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

        # Add to log
        logger.info("Step 5 - Distribution Statistics Calculation")
        logger.info(f"Mean tokens: {mean_tokens:.2f}")
        logger.info(f"5th percentile threshold: {percentile_5th:.2f} tokens")
        logger.info(f"95th percentile threshold: {percentile_95th:.2f} tokens")
        logger.info(f"Total chunks: {total_chunks}")
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
            logger.info("Merge process for chunks under 5th percentile")
            final_chunks = merge_undersized_chunks(
                chunks=final_chunks,
                min_token_threshold=percentile_5th,
                max_tokens=input_data.max_tokens,
                model=input_data.model,
            )

        # STEP 7: Split oversized chunks
        logger.info("Step 7 - Split Oversized Chunks")
        oversized_indices = [
            i
            for i, chunk in enumerate(final_chunks)
            if chunk["token_count"] > input_data.max_tokens
        ]

        if oversized_indices:
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

        logger.info(
            f"After splitting: {len(final_chunks)} total chunks (increased from {chunks_before_split} to {len(final_chunks)}, +{len(final_chunks) - chunks_before_split})"
        )

        logger.info("Step 8 - Semantic Chunking Statistics after processing")

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
                processing_time=processing_time,
            ),
        )

        # Schedule cleanup after response is sent
        if background_tasks:
            background_tasks.add_task(lambda: torch.cuda.empty_cache() if torch.cuda.is_available() else None)

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

        # Clean GPU memory in case of error too
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return a safe error message without exposing system details
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the chunking request: {error_type}. Please check the logs for more information.",
        )
