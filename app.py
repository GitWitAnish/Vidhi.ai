"""
HuggingFace Spaces entry point.
Launches the FastAPI backend using uvicorn on port 7860 (HF Spaces default).
Uses Gradio SDK (free tier) but runs FastAPI instead.
"""

import subprocess
import sys
import os

# Launch uvicorn pointing to the FastAPI app in the backend directory
subprocess.run(
    [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "7860",
    ],
    cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"),
)
