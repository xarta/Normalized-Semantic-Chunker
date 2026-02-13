#!/usr/bin/env python3
"""
Normalised Semantic Chunker — health checker and test tool.

Checks the deployed Normalised Semantic Chunker service, reads endpoint
from .env, and runs test requests.

Usage:
    python3 tools/check_service.py              # Health check
    python3 tools/check_service.py --test       # Test chunking
    python3 tools/check_service.py --json       # JSON output
    python3 tools/check_service.py --all        # Everything

Zero external dependencies — stdlib only.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ===================================================================
# .env loader
# ===================================================================

def load_env(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Load key=value pairs from .env file."""
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"
    env: Dict[str, str] = {}
    if not env_path.exists():
        return env
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            env[key] = value
    return env


# ===================================================================
# HTTP helpers
# ===================================================================

def http_get(url: str, timeout: int = 10) -> Tuple[int, str]:
    """GET request, returns (status_code, body)."""
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


def http_post_file(
    url: str, filename: str, content: bytes, timeout: int = 60,
) -> Tuple[int, str]:
    """POST multipart/form-data file upload, returns (status_code, body)."""
    boundary = "----ChunkerTestBoundary9876543210"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/plain\r\n"
        f"\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


# ===================================================================
# Health check
# ===================================================================

def check_health(base_url: str) -> Dict[str, Any]:
    """Check Normalised Semantic Chunker health."""
    start = time.monotonic()
    status, body = http_get(f"{base_url}/")
    latency_ms = (time.monotonic() - start) * 1000

    result: Dict[str, Any] = {
        "service": "Normalised Semantic Chunker",
        "endpoint": base_url,
        "latency_ms": round(latency_ms, 1),
    }

    if status == 200:
        try:
            data = json.loads(body)
            result["status"] = data.get("status", "unknown")
            result["embedding_backend"] = data.get("embedding_backend")
            result["embedding_model"] = data.get("embedding_model")
            result["version"] = data.get("version")
        except json.JSONDecodeError:
            result["status"] = "error"
            result["error"] = "Invalid JSON response"
    else:
        result["status"] = "down"
        result["error"] = body[:200] if body else f"HTTP {status}"

    return result


# ===================================================================
# Test request
# ===================================================================

TEST_TEXT = (
    "Text splitting in LangChain is a critical feature that enables "
    "the division of large texts into smaller, manageable segments. "
    "This capability is essential for improving comprehension and "
    "processing efficiency, particularly in tasks that require "
    "detailed analysis or the extraction of specific contexts.\n\n"
    "ChatGPT, developed by OpenAI, represents a significant "
    "advancement in natural language processing technologies. "
    "As a conversational AI model, ChatGPT is capable of "
    "understanding and generating human-like text, facilitating "
    "dynamic and engaging interactions."
)


def test_chunking(base_url: str) -> Dict[str, Any]:
    """Send a test request to the Normalised Semantic Chunker."""
    url = f"{base_url}/normalized_semantic_chunker/?max_tokens=200"
    content = TEST_TEXT.encode("utf-8")

    start = time.monotonic()
    status, body = http_post_file(url, "test.txt", content, timeout=60)
    elapsed = time.monotonic() - start

    result: Dict[str, Any] = {
        "service": "Normalised Semantic Chunker",
        "endpoint": url,
        "elapsed_seconds": round(elapsed, 2),
    }

    if status == 200:
        try:
            data = json.loads(body)
            meta = data.get("metadata", {})
            chunks = data.get("chunks", [])
            result["status"] = "ok"
            result["n_chunks"] = meta.get("n_chunks", len(chunks))
            result["avg_tokens"] = meta.get("avg_tokens")
            result["processing_time"] = meta.get("processing_time")
            result["embedder_model"] = meta.get("embedder_model")
            if chunks:
                first = chunks[0]
                text_preview = first.get("text", "")[:100]
                result["first_chunk_preview"] = (
                    text_preview + "..." if len(first.get("text", "")) > 100
                    else text_preview
                )
        except json.JSONDecodeError:
            result["status"] = "error"
            result["error"] = "Invalid JSON response"
    else:
        result["status"] = "error"
        result["http_status"] = status
        result["error"] = body[:300] if body else "empty response"

    return result


# ===================================================================
# Output formatting
# ===================================================================

STATUS_ICONS = {
    "healthy": "[OK]",
    "ok": "[OK]",
    "unhealthy": "[FAIL]",
    "down": "[FAIL]",
    "error": "[FAIL]",
    "skip": "[SKIP]",
}


def format_health(result: Dict[str, Any], as_json: bool = False) -> str:
    """Format health check result."""
    if as_json:
        return json.dumps(result, indent=2)

    icon = STATUS_ICONS.get(result.get("status", ""), "[??]")
    svc = result.get("service", "Unknown")
    line = f"  {icon:8s} {svc}"
    model = result.get("embedding_model")
    if model:
        line += f"  ({model})"
    latency = result.get("latency_ms")
    if latency is not None:
        line += f"  [{latency:.0f}ms]"

    lines = [
        "Normalised Semantic Chunker Health Check",
        "=" * 40,
        "",
        line,
    ]
    if result.get("error"):
        lines.append(f"           error: {result['error']}")

    lines.append("")
    up = 1 if result.get("status") in ("healthy", "ok") else 0
    down = 1 - up
    lines.append(f"  {up} up, {down} down")
    lines.append(f"  Overall: {'HEALTHY' if up else 'UNHEALTHY'}")
    return "\n".join(lines)


def format_test(result: Dict[str, Any], as_json: bool = False) -> str:
    """Format test result."""
    if as_json:
        return json.dumps(result, indent=2)

    status = result.get("status", "unknown")
    icon = STATUS_ICONS.get(status, "[??]")
    elapsed = result.get("elapsed_seconds", 0)

    lines = [f"  {icon:8s} Normalised Semantic Chunker test ({elapsed:.1f}s)"]

    if status == "ok":
        if "n_chunks" in result:
            lines.append(f"           chunks: {result['n_chunks']}")
        if result.get("avg_tokens"):
            lines.append(f"           avg_tokens: {result['avg_tokens']:.1f}")
        if result.get("embedder_model"):
            lines.append(f"           model: {result['embedder_model']}")
        if result.get("first_chunk_preview"):
            lines.append(f"           preview: {result['first_chunk_preview']}")
    else:
        if result.get("error"):
            lines.append(f"           error: {result['error'][:200]}")

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Normalised Semantic Chunker service health and run test requests.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Send a test chunking request",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all checks: health and test",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--env", type=Path, default=None,
        help="Path to .env file (default: project root .env)",
    )
    args = parser.parse_args()

    env = load_env(args.env)
    base_url = env.get("NORMALISED_CHUNKER_URL", "")

    if not base_url:
        print("Error: NORMALISED_CHUNKER_URL not set in .env", file=sys.stderr)
        print("Copy .env.example to .env and configure the endpoint.", file=sys.stderr)
        return 1

    do_test = args.test or args.all
    outputs = []
    any_failure = False

    # Health check (always)
    health = check_health(base_url)
    outputs.append(format_health(health, args.json))
    if health.get("status") not in ("healthy", "ok"):
        any_failure = True

    # Test request
    if do_test:
        outputs.append("")
        outputs.append("Test: Normalised Semantic Chunker")
        outputs.append("-" * 40)
        result = test_chunking(base_url)
        outputs.append(format_test(result, args.json))
        if result.get("status") != "ok":
            any_failure = True

    print("\n".join(outputs))
    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
