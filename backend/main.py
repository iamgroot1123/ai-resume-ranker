"""
AI Resume Ranker — FastAPI Backend
-----------------------------------
• POST /api/rank     — rank uploaded documents against a query
• GET  /api/health   — liveness check
• GET  /*            — serve the React SPA (production build)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.utils import load_model_once, rank_resumes, analyze_applicant

# ---------------------------------------------------------------------------
# Lifespan — load heavy resources once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Loading SBERT model…")
    app.state.model = load_model_once()
    if app.state.model is None:
        print("[WARN] SBERT model failed to load. Ranking will be unavailable.")
    else:
        print("[INFO] SBERT model ready.")
    yield
    # Nothing to clean up


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Resume Ranker API", version="2.0.0", lifespan=lifespan)

# CORS — allow the Vite dev server and any deployed origin
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tightened via env in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health(request: Request):
    model_ok = getattr(request.app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": model_ok}


@app.post("/api/rank")
async def rank_endpoint(
    request: Request,
    # Query / job description
    job_desc_text: str = Form(""),
    job_desc_file: Optional[UploadFile] = File(None),
    # Ranking config
    keywords: str = Form(""),
    top_n: int = Form(5),
    use_llm: bool = Form(False),
    api_key: str = Form(""),
    # Model to use
    llm_model: str = Form("gpt-3.5-turbo-1106"),
    # Resume files
    resumes: List[UploadFile] = File(...),
):
    """
    Rank uploaded resumes/documents against a job description.

    The OpenAI API key (if provided) is used only for this request and
    is never logged, persisted, or stored anywhere on the server.
    """
    # --- Resolve job description ---
    final_jd = job_desc_text.strip()
    if job_desc_file and job_desc_file.size and job_desc_file.size > 0:
        jd_bytes = await job_desc_file.read()
        final_jd = jd_bytes.decode("utf-8", errors="ignore").strip()

    if not final_jd:
        raise HTTPException(status_code=422, detail="A job description is required.")

    # --- Validate model ---
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The semantic model is not loaded. Please try again in a moment.",
        )

    # --- Read resume files into memory ---
    uploaded: list[dict] = []
    for resume_file in resumes:
        if resume_file.size == 0:
            continue
        content = await resume_file.read()
        uploaded.append({"name": resume_file.filename, "bytes": content})

    if not uploaded:
        raise HTTPException(status_code=422, detail="Please upload at least one resume.")

    # --- Validate LLM requirements ---
    if use_llm and not api_key.strip():
        raise HTTPException(
            status_code=422,
            detail="An OpenAI API key is required for LLM ranking. Please provide one or switch to SBERT-only mode.",
        )

    # --- Run ranking pipeline ---
    results, error = rank_resumes(
        job_desc_text=final_jd,
        keywords=keywords,
        top_n=top_n,
        uploaded_resumes=uploaded,
        model=model,
        api_key=api_key.strip(),
        use_llm=use_llm,
        llm_model=llm_model,
    )

    if error:
        raise HTTPException(status_code=400, detail=error)

    return JSONResponse(
        content={
            "candidates": results,
            "total_uploaded": len(uploaded),
            "total_returned": len(results),
            "scoring_mode": "llm+sbert" if use_llm else "sbert",
        }
    )


@app.post("/api/analyze")
async def analyze_endpoint(
    request: Request,
    # Job description
    job_desc_text: str = Form(""),
    job_desc_file: Optional[UploadFile] = File(None),
    # LLM config
    api_key: str = Form(""),
    llm_model: str = Form("gpt-3.5-turbo-1106"),
    # Single resume file
    resume: UploadFile = File(...),
):
    """
    Analyze a single resume against a job description (Applicant Mode).
    Returns match score, summary, strengths, gaps, and actionable suggestions.
    """
    # --- Resolve job description ---
    final_jd = job_desc_text.strip()
    if job_desc_file and job_desc_file.size and job_desc_file.size > 0:
        jd_bytes = await job_desc_file.read()
        final_jd = jd_bytes.decode("utf-8", errors="ignore").strip()

    if not final_jd:
        raise HTTPException(status_code=422, detail="A job description is required.")

    # --- Validate SBERT model ---
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The semantic model is not loaded. Please try again in a moment.",
        )

    # --- Read resume file ---
    if resume.size == 0:
        raise HTTPException(status_code=422, detail="Resume file is empty.")
    content = await resume.read()
    resume_dict = {"name": resume.filename, "bytes": content}

    # --- Run applicant analysis ---
    result, error = analyze_applicant(
        resume_file=resume_dict,
        job_desc_text=final_jd,
        sbert_model=model,
        api_key=api_key.strip(),
        llm_model=llm_model,
    )

    if error:
        raise HTTPException(status_code=400, detail=error)

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# SPA Static File Serving  (production — after npm run build)
# ---------------------------------------------------------------------------
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    # Serve compiled assets
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Catch-all — serve the React SPA for all non-API routes."""
        index_file = FRONTEND_DIST / "index.html"
        return FileResponse(str(index_file))
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": "Frontend not built. Run `npm run build` inside the `frontend/` directory."
        }
