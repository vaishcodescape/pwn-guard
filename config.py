"""
Project configuration: paths and env-based settings.
All paths are relative to the project root (directory containing this file).
"""

import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# Paths
PROJECT_ROOT = _env("PROJECT_ROOT", _PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# API
API_VERSION = _env("API_VERSION", "1.0.0")
HOST = _env("HOST", "0.0.0.0")
PORT = int(_env("PORT", "8000"))
