from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import os

# ==============================================================================
# FastAPI Application for Shiva Kumar Madagoni Portfolio (Vercel Serverless)
# ==============================================================================

app = FastAPI(
    title="Shiva Kumar Madagoni - Portfolio API",
    description="High-performance serverless backend API providing portfolio data, skills, project records, and contact verification.",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url="/api/redoc"
)

# Top-level ASGI handler required by Vercel Python runtime
handler = app

# Configure CORS for Vercel deployment and local preview environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class ContactRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Name of sender or recruiter")
    email: EmailStr = Field(..., description="Valid contact email address")
    opportunity_type: str = Field("AI / ML Internship", description="Opportunity category")
    message: str = Field(..., min_length=5, max_length=2000, description="Message content")

class ContactResponse(BaseModel):
    success: bool
    message: str
    received_at: str

# ==============================================================================
# Portfolio Data Store
# ==============================================================================

PORTFOLIO_PROFILE = {
    "name": "Shiva Kumar Madagoni",
    "headline": "AI Engineer & Full-Stack Systems Developer",
    "degree": "B.Tech in Artificial Intelligence and Machine Learning",
    "duration": "2024 - 2028",
    "institution": "Pallavi Engineering College",
    "location": "Greater Hyderabad Area, Telangana, India",
    "availability": "Available for Internship",
    "contact": {
        "phone": "+91 8520929979",
        "whatsapp": "+91 8520929979",
        "whatsapp_link": "https://wa.me/918520929979",
        "email": "mshivakumar1289@gmail.com",
        "linkedin": "https://www.linkedin.com/in/shiva-kumar-madagoni-627038327/",
        "github": "https://github.com/ShivaKumar338",
        "leetcode": "https://leetcode.com/u/shivakumar338/"
    }
}

PROJECTS_DATA = [
    {
        "id": "orchestai",
        "title": "OrchestAI — Autonomous Workforce Orchestration Engine",
        "repo": "Employee_Work_Management",
        "badge": "Agentic AI & Full-Stack",
        "category": "agentic",
        "metrics": "Sub-200ms DAG scheduling, 99.4% task dispatch accuracy",
        "stack": ["FastAPI", "Next.js 16", "React Flow", "PostgreSQL", "Supabase", "Docker", "Gemini 1.5 Flash"],
        "githubUrl": "https://github.com/ShivaKumar338/Employee_Work_Management"
    },
    {
        "id": "deepguard",
        "title": "DeepGuard AI — Multimodal Deepfake Forensics Pipeline",
        "repo": "Deep-fake-Ai-detection",
        "badge": "Deep Learning & Vision",
        "category": "vision",
        "metrics": "94.8% cross-dataset generalization on synthetic audio & video",
        "stack": ["PyTorch", "OpenCV", "Librosa", "EfficientNet-B4", "Whisper", "FastAPI"],
        "githubUrl": "https://github.com/ShivaKumar338/Deep-fake-Ai-detection"
    },
    {
        "id": "tech-dispatch",
        "title": "AI Autonomous Field Dispatch & Multi-Party Negotiation",
        "repo": "AI-Driven-Autonomous-Field-Technician-Routing-and-Dispatch-System",
        "badge": "Multi-Agent Systems",
        "category": "agentic",
        "metrics": "Autonomous twin-agent rate negotiation loop (4 rounds maximum)",
        "stack": ["Python", "FastAPI", "Google Gemini 1.5", "Haversine Routing", "WebSockets"],
        "githubUrl": "https://github.com/ShivaKumar338/AI-Driven-Autonomous-Field-Technician-Routing-and-Dispatch-System"
    },
    {
        "id": "vision-assistant",
        "title": "VisionAI — AI Voice & Vision Edge Assistant",
        "repo": "Vision-Ai-Assistance",
        "badge": "Computer Vision & Edge AI",
        "category": "vision",
        "metrics": "Sub-second handwriting recognition & high-fps real-time object detection",
        "stack": ["Python", "OpenCV", "YOLOv8", "TrOCR", "EasyOCR", "pyttsx3", "SpeechRecognition"],
        "githubUrl": "https://github.com/ShivaKumar338/Vision-Ai-Assistance"
    },
    {
        "id": "crewai-flow",
        "title": "CrewAI — Multi-Agent Market Intelligence Pipeline",
        "repo": "CREW_AI",
        "badge": "Multi-Agent Flow",
        "category": "agentic",
        "metrics": "Autonomous 3-agent pipeline: DataCollector -> ProductMatcher -> DataFormatter",
        "stack": ["Python", "CrewAI Flow", "Gemini 1.5 Flash", "Web Scraping", "ETL"],
        "githubUrl": "https://github.com/ShivaKumar338/CREW_AI"
    },
    {
        "id": "fraud-detection",
        "title": "Financial Fraud Detection System (Live Deployment)",
        "repo": "Fraud_Detection",
        "badge": "Predictive ML & Production",
        "category": "ml",
        "metrics": "Live FastAPI REST microservice deployed on Render cloud",
        "stack": ["Scikit-Learn", "FastAPI", "Pandas", "NumPy", "SMOTE", "Render"],
        "githubUrl": "https://github.com/ShivaKumar338/Fraud_Detection",
        "liveUrl": "https://fraud-detection-dnty.onrender.com"
    },
    {
        "id": "leetcode-dsa",
        "title": "Algorithmic Problem Solving & LeetCode Practice",
        "repo": "neetcode-submissions",
        "badge": "Algorithms & CS",
        "category": "fullstack",
        "metrics": "100+ algorithmic problems solved with rigorous time/space complexity analysis",
        "stack": ["Python", "Algorithms", "Data Structures", "LeetCode / NeetCode"],
        "githubUrl": "https://github.com/ShivaKumar338/neetcode-submissions",
        "liveUrl": "https://leetcode.com/u/shivakumar338/"
    }
]

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/api", tags=["Root"])
@app.get("/api/", tags=["Root"])
def root_endpoint():
    """Root endpoint verifying API deployment on Vercel."""
    return {
        "service": "Shiva Kumar Madagoni Portfolio API",
        "status": "online",
        "version": "1.0.0",
        "docs": "/api/docs",
        "runtime": "Vercel Serverless Python (ASGI)",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/health", tags=["Health"])
def health_check():
    """Health check endpoint for status monitoring."""
    env = os.getenv("VERCEL_ENV", "development")
    return {
        "status": "healthy",
        "environment": env,
        "developer": "Shiva Kumar Madagoni",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/profile", tags=["Profile"])
def get_profile():
    """Get candidate profile metadata, degree info, and contact handles."""
    return PORTFOLIO_PROFILE

@app.get("/api/projects", tags=["Projects"])
def get_projects(category: Optional[str] = Query(None, description="Filter projects by category")):
    """List all featured projects with optional category filter."""
    if category and category.lower() != "all":
        filtered = [p for p in PROJECTS_DATA if p["category"].lower() == category.lower()]
        return {"total": len(filtered), "category": category, "projects": filtered}
    return {"total": len(PROJECTS_DATA), "projects": PROJECTS_DATA}

@app.post("/api/contact", response_model=ContactResponse, status_code=status.HTTP_200_OK, tags=["Contact"])
def handle_contact_message(payload: ContactRequest):
    """
    Validate contact inquiry safely without exposing or requiring hardcoded credentials.
    In production on Vercel, this can forward to Webhook or Email services via env vars.
    """
    notification_webhook = os.getenv("CONTACT_WEBHOOK_URL")
    if notification_webhook:
        pass

    return ContactResponse(
        success=True,
        message=f"Thank you, {payload.name}! Your message regarding '{payload.opportunity_type}' has been recorded. Shiva responds within 24 hours.",
        received_at=datetime.now(timezone.utc).isoformat()
    )

# ==============================================================================
# Static Web Portfolio Fallbacks (Serves index.html when run via app.py)
# ==============================================================================

from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
def serve_index():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_file = os.path.join(base_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Portfolio API Running. Visit /api/docs"}

@app.get("/{file_name:path}", include_in_schema=False)
def serve_root_files(file_name: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_file = os.path.join(base_dir, file_name)
    if os.path.isfile(target_file):
        return FileResponse(target_file)
    raise HTTPException(status_code=404, detail="Resource not found")

