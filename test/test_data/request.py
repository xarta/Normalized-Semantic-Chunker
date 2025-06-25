import requests
import json
import os

api_url = "http://localhost:8000/normalized_semantic_chunker/"
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "alice_in_wonderland.txt")
max_tokens_per_chunk = 500

try:
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "text/plain")}
        params = {"max_tokens": max_tokens_per_chunk}

        response = requests.post(api_url, files=files, params=params)
        response.raise_for_status()

        result = response.json()

        print(
            f"Successfully chunked document into {result['metadata']['n_chunks']} chunks."
        )
        output_file = "response.json"
        with open(output_file, "w", encoding="utf-8") as outfile:
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
