"""
app.py - Local ASGI Development Server for Shiva Kumar Madagoni Portfolio
Run locally via: python app.py or uvicorn app:app --reload
"""
import sys
import os

# Ensure workspace root and api directory are in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.index import app, handler
from fastapi.staticfiles import StaticFiles

# Mount static frontend files for local development previews
static_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(static_dir, "index.html")):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

application = app

__all__ = ["app", "application", "handler"]

if __name__ == "__main__":
    import uvicorn
    print("============================================================")
    print("🚀 Shiva Kumar Madagoni - Portfolio Full-Stack Server")
    print("📡 Local Server: http://localhost:8000")
    print("📖 Swagger Docs: http://localhost:8000/api/docs")
    print("============================================================")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
