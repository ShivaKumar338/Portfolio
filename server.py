import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

PORT = int(os.environ.get("PORT", "8000"))
SITE_DIR = Path(__file__).resolve().parent
SITE_FILES = {
    "index.html": SITE_DIR / "index.html",
    "style.css": SITE_DIR / "style.css",
    "script.js": SITE_DIR / "script.js",
    "projects-data.js": SITE_DIR / "projects-data.js",
    "Shiva.png": SITE_DIR / "Shiva.png",
}

app = FastAPI(
    title="Shiva Kumar Madagoni Portfolio",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(SITE_FILES["index.html"])


@app.get("/{asset_name}", include_in_schema=False)
async def site_asset(asset_name: str) -> FileResponse:
    asset = SITE_FILES.get(asset_name)
    if asset is None:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(asset)


if __name__ == "__main__":
    print(f"Portfolio server running at http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
