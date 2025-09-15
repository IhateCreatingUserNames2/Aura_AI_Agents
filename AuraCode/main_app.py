# main_app.py
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify critical environment variables
if not os.getenv("OPENROUTER_API_KEY"):
    print("WARNING: OPENROUTER_API_KEY environment variable is not set!")
    print("The agents will fail to make LLM calls without this.")
    print("Please set it in your .env file or environment.")

if not os.getenv("JWT_SECRET_KEY"):
    print("WARNING: Using default JWT secret key. This is insecure for production!")
    print("Please set JWT_SECRET_KEY in your .env file.")

# Import the FastAPI app from routes
from api.routes import app

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    print(f"Starting Aura Multi-Agent API on {host}:{port}")
    print(f"API Documentation will be available at http://localhost:{port}/docs")
    print(f"Frontend (if present) will be available at http://localhost:{port}/")

    uvicorn.run(
        "api.routes:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )