"""
app.py - Top-level Application Entrypoint for Vercel Deployment & Local ASGI Server
Exports top-level 'app', 'application', and 'handler' variables as required by Vercel's Python runtime.
"""
import sys
import os

# Ensure workspace root and api directory are in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.index import app, handler

# Top-level exports required by Vercel Python runtime
application = app

__all__ = ["app", "application", "handler"]

if __name__ == "__main__":
    import uvicorn
    print("============================================================")
    print("🚀 Shiva Kumar Madagoni - Portfolio FastAPI Backend")
    print("📡 Local Server: http://localhost:8000")
    print("📖 Swagger Docs: http://localhost:8000/api/docs")
    print("============================================================")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
