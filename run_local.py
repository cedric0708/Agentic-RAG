"""로컬 API 실행: python run_local.py (http://127.0.0.1:8001)."""

from pathlib import Path

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        app_dir=str(Path(__file__).resolve().parent),
        host="127.0.0.1",
        port=8001,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent)],
    )
