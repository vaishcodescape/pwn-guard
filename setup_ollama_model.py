#!/usr/bin/env python3
"""
Setup script to download sarah.gguf model from Hugging Face and import it into Ollama.
Uses retries, longer timeouts, and a direct-URL fallback so the model loads reliably.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

# Config via env (default: Het0456/sarah, sarah.gguf)
HF_REPO = os.environ.get("SARAH_HF_REPO", "Het0456/sarah")
HF_FILENAME = os.environ.get("SARAH_HF_FILENAME", "sarah.gguf")
MODEL_NAME = os.environ.get("OLLAMA_SARAH_MODEL", "sarah")
DOWNLOAD_RETRIES = int(os.environ.get("SARAH_DOWNLOAD_RETRIES", "5"))
RETRY_DELAY_BASE = int(os.environ.get("SARAH_RETRY_DELAY", "15"))  # seconds


def wait_for_ollama(max_attempts=30):
    """Wait for Ollama server to be ready."""
    print("Waiting for Ollama to be ready...")
    for i in range(max_attempts):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✓ Ollama is ready!")
                return True
        except Exception:
            pass
        if i < max_attempts - 1:
            print(f"  Attempt {i+1}/{max_attempts}...")
            time.sleep(2)
    print("✗ Ollama failed to start")
    return False


def check_model_exists(model_name=None):
    """Check if model already exists in Ollama."""
    model_name = model_name or MODEL_NAME
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return model_name in result.stdout
    except Exception:
        return False


def _download_via_huggingface_hub(model_dir: Path):
    """Download using huggingface_hub with retries and resume (public repo, no token)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    kwargs = dict(
        repo_id=HF_REPO,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            print(f"  Attempt {attempt}/{DOWNLOAD_RETRIES} (huggingface_hub)...")
            path = hf_hub_download(
                filename=HF_FILENAME,
                local_dir=str(model_dir),
                **kwargs,
            )
            print(f"✓ Downloaded model to {path}")
            return str(path)
        except Exception as e:
            print(f"  hf_hub_download failed: {e}")
            if attempt < DOWNLOAD_RETRIES:
                delay = RETRY_DELAY_BASE * attempt
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            print(f"  Attempt {attempt}/{DOWNLOAD_RETRIES} (snapshot_download)...")
            repo_path = snapshot_download(
                local_dir=str(model_dir / "sarah_repo"),
                **kwargs,
            )
            gguf_files = list(Path(repo_path).rglob("*.gguf"))
            if gguf_files:
                path = str(gguf_files[0])
                print(f"✓ Found GGUF: {path}")
                return path
        except Exception as e:
            print(f"  snapshot_download failed: {e}")
            if attempt < DOWNLOAD_RETRIES:
                delay = RETRY_DELAY_BASE * attempt
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
    return None


def _download_via_direct_url(model_dir: Path):
    """Fallback: stream file from HF resolve URL (CDN often works when API returns 502)."""
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_FILENAME}"
    out_path = model_dir / HF_FILENAME
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            print(f"  Attempt {attempt}/{DOWNLOAD_RETRIES} (direct URL)...")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or 0
                with open(out_path, "wb") as f:
                    for i, chunk in enumerate(r.iter_content(chunk_size=2**20)):
                        if chunk:
                            f.write(chunk)
                        if total and (i % 10 == 0):
                            done = f.tell()
                            pct = (100 * done // total) if total else 0
                            print(f"  Downloaded {done // (1024*1024)} MB ({pct}%)")
            if out_path.is_file() and out_path.stat().st_size > 0:
                print(f"✓ Downloaded to {out_path}")
                return str(out_path)
        except Exception as e:
            print(f"  Direct download failed: {e}")
            if out_path.exists():
                out_path.unlink()
            if attempt < DOWNLOAD_RETRIES:
                delay = RETRY_DELAY_BASE * attempt
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
    return None


def download_model():
    """Download model from Hugging Face with retries and fallbacks."""
    model_dir = Path("/tmp/ollama_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_FILENAME} from Hugging Face ({HF_REPO})...")

    path = _download_via_huggingface_hub(model_dir)
    if path:
        return path
    path = _download_via_direct_url(model_dir)
    if path:
        return path

    raise RuntimeError(
        f"Could not download {HF_FILENAME} after all retries. "
        "Check network and https://huggingface.co/" + HF_REPO
    )


def create_ollama_model(gguf_path: str, model_name: str = None):
    """Create Ollama model from GGUF file."""
    model_name = model_name or MODEL_NAME
    print(f"Creating {model_name} model in Ollama...")

    modelfile_content = f"""FROM {gguf_path}
PARAMETER stop <|start_header_id|>
PARAMETER stop <|end_header_id|>
PARAMETER stop <|eot_id|>
"""
    modelfile_path = f"/tmp/Modelfile.{model_name}"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode == 0:
        print(f"✓ {model_name} model created successfully")
        return True
    print(f"✗ Failed to create model: {result.stderr}")
    return False


def main():
    """Main setup: wait for Ollama, download Sarah, create model."""
    print("=== Setting up Ollama with Sarah Model ===\n")

    if not wait_for_ollama():
        sys.exit(1)

    if check_model_exists():
        print(f"✓ {MODEL_NAME} model already exists")
        return

    try:
        gguf_path = download_model()
        if create_ollama_model(gguf_path):
            print("\n✓ Setup complete!")
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
