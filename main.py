"""
TalentLens AI — Multi-Agent Resume Intelligence Backend v3.0
FastAPI + Sentence Transformers + Multi-Agent Orchestration
Production-ready: API auth, rate limiting, batch processing,
candidate store, async jobs, webhook support.
"""

import logging
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import os

from agents.orchestrator import Orchestrator
from agents.parser_agent import ParserAgent
from agents.normalizer_agent import NormalizerAgent
from agents.matcher_agent import MatcherAgent
from models.schemas import MatchRequest
from middleware.auth import require_api_key, get_demo_keys
from store.candidate_store import CandidateStore, JobQueue

# ── Structured Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("talentlens")


# ── Lifespan: Initialize all agents & stores ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing TalentLens AI v3.0 agents…")
    app.state.parser     = ParserAgent()
    app.state.normalizer = NormalizerAgent()
    app.state.matcher    = MatcherAgent()
    app.state.orchestrator = Orchestrator(
        app.state.parser, app.state.normalizer, app.state.matcher
    )
    app.state.candidate_store = CandidateStore(ttl_seconds=86400)
    app.state.job_queue       = JobQueue()
    logger.info("All agents ready. TalentLens AI v3.0 is live.")
    yield
    logger.info("Shutting down TalentLens AI.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TalentLens AI",
    description="Multi-Agent Resume Intelligence API — semantic embeddings, rule-based inference, and AI-powered hiring insights.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception Handlers ────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    msg = errors[0]["msg"] if errors else "Invalid request data."
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "message": msg, "details": errors}
    )


# ── Timing Middleware ─────────────────────────────────────────────────────────
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = round((time.time() - start) * 1000)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    response.headers["X-TalentLens-Version"] = "3.0.0"
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed_ms}ms)")
    return response


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Frontend"])
async def serve_index():
    """Serve the TalentLens AI frontend dashboard."""
    if os.path.exists("index.html"):
        return FileResponse("index.html", media_type="text/html")
    return JSONResponse(content={
        "product": "TalentLens AI", "version": "3.0.0", "status": "running",
        "docs": "/docs", "health": "/api/v1/health",
    })


# ── Health & Status ───────────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["Status"])
async def health(request: Request):
    """Detailed health check — confirms all agents are loaded and ready."""
    orch    = getattr(request.app.state, "orchestrator", None)
    matcher = getattr(request.app.state, "matcher", None)
    store   = getattr(request.app.state, "candidate_store", None)
    sbert_loaded = bool(matcher and matcher.model is not None)
    store_stats  = store.stats() if store else {}
    return {
        "status":         "healthy",
        "version":        "3.0.0",
        "agents_loaded":  orch is not None,
        "sbert_available": sbert_loaded,
        "match_method":   "semantic (SBERT)" if sbert_loaded else "exact + heuristic",
        "candidate_store": store_stats,
        "demo_keys":      get_demo_keys(),
    }


# ── Core Analysis ─────────────────────────────────────────────────────────────
@app.post("/api/v1/analyze", tags=["Analysis"])
async def analyze(
    request_body: MatchRequest,
    request: Request,
    _auth=Depends(require_api_key),
):
    """
    Full 7-stage multi-agent analysis pipeline.
    Input: resume_text, job_description, candidate_name.
    Returns: match score, skill gaps, AI insight, pipeline logs.
    """
    start = time.time()
    orchestrator: Orchestrator        = request.app.state.orchestrator
    store: CandidateStore             = request.app.state.candidate_store
    logger.info(f"Analysis requested for: {request_body.candidate_name}")

    try:
        result = await orchestrator.full_analysis(
            resume_text=request_body.resume_text,
            job_description=request_body.job_description,
            candidate_name=request_body.candidate_name or "Candidate",
            years_exp=request_body.expected_years or 0,
        )
        # Store and tag with candidate ID
        candidate_id = store.add(result)
        result["candidate_id"] = candidate_id
        result["api_processing_ms"] = round((time.time() - start) * 1000)

        # Fire webhook asynchronously if provided
        if request_body.webhook_url:
            asyncio.create_task(
                orchestrator._fire_webhook(request_body.webhook_url, "sync", result)
            )

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Analysis pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "error": "pipeline_error",
            "message": f"Analysis failed: {str(e)}",
            "hint": "Check that resume_text and job_description are non-empty.",
        })


@app.post("/api/v1/match", tags=["Analysis"])
async def match_alias(request_body: MatchRequest, request: Request, _auth=Depends(require_api_key)):
    """Backwards-compatible alias for /api/v1/analyze."""
    return await analyze(request_body, request, _auth)


# ── Async Job Endpoint ────────────────────────────────────────────────────────
@app.post("/api/v1/analyze/async", tags=["Analysis"])
async def analyze_async(
    request_body: MatchRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    _auth=Depends(require_api_key),
):
    """
    Submit an analysis job for background processing.
    Returns a job_id immediately. Poll /api/v1/jobs/{job_id} for status.
    Optionally provide webhook_url to receive a callback when done.
    """
    job_queue: JobQueue         = request.app.state.job_queue
    store: CandidateStore       = request.app.state.candidate_store
    orch: Orchestrator          = request.app.state.orchestrator

    job_id = job_queue.create(description=f"Analysis for {request_body.candidate_name}")
    background_tasks.add_task(
        orch.async_analyze,
        request_body.resume_text,
        request_body.job_description,
        request_body.candidate_name or "Candidate",
        job_queue,
        store,
        job_id,
        request_body.webhook_url,
    )
    return JSONResponse(status_code=202, content={
        "job_id": job_id,
        "status": "pending",
        "poll_url": f"/api/v1/jobs/{job_id}",
        "message": "Analysis job submitted. Poll poll_url for results.",
    })


@app.get("/api/v1/jobs/{job_id}", tags=["Analysis"])
async def get_job_status(job_id: str, request: Request, _auth=Depends(require_api_key)):
    """Check the status of an async analysis job."""
    job_queue: JobQueue = request.app.state.job_queue
    job = job_queue.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "job_not_found", "job_id": job_id})
    return JSONResponse(content=job)


# ── Resume Parsing ────────────────────────────────────────────────────────────
@app.post("/api/v1/parse", tags=["Resume"])
async def parse_resume(
    request: Request,
    file: UploadFile = File(...),
    _auth=Depends(require_api_key),
):
    """Parse a resume file (PDF, DOCX, TXT) and return structured data."""
    start = time.time()
    parser: ParserAgent = request.app.state.parser

    allowed = (".pdf", ".docx", ".txt")
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(status_code=422, detail={
            "error": "invalid_file_type",
            "message": "Only PDF, DOCX, or TXT files are supported.",
            "allowed": list(allowed),
        })

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail={
            "error": "file_too_large", "message": "File must be under 10MB."
        })

    try:
        result = await parser.parse(content, file.filename)
        result["processing_time_ms"] = round((time.time() - start) * 1000)
        result["agent"] = "ParserAgent v3.0"
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Parse error for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "parse_error", "message": str(e)})


@app.post("/api/v1/parse/batch", tags=["Resume"])
async def batch_parse_resumes(
    request: Request,
    files: List[UploadFile] = File(...),
    _auth=Depends(require_api_key),
):
    """
    Batch parse up to 10 resume files in parallel.
    Returns structured data for each file plus aggregate timing.
    """
    start = time.time()
    if len(files) > 10:
        raise HTTPException(status_code=422, detail={
            "error": "too_many_files", "message": "Batch limit is 10 files."
        })

    parser: ParserAgent = request.app.state.parser

    async def _parse_one(f: UploadFile) -> Dict:
        content = await f.read()
        try:
            result = await parser.parse(content, f.filename)
            result["status"] = "parsed"
            return result
        except Exception as e:
            return {"filename": f.filename, "status": "failed", "error": str(e)}

    results = await asyncio.gather(*[_parse_one(f) for f in files])
    succeeded = sum(1 for r in results if r.get("status") == "parsed")

    return JSONResponse(content={
        "total":              len(files),
        "succeeded":          succeeded,
        "failed":             len(files) - succeeded,
        "results":            list(results),
        "processing_time_ms": round((time.time() - start) * 1000),
    })


# ── Candidate Store ───────────────────────────────────────────────────────────
@app.get("/api/v1/candidates", tags=["Candidates"])
async def list_candidates(request: Request, _auth=Depends(require_api_key)):
    """List all stored candidates with summary data."""
    store: CandidateStore = request.app.state.candidate_store
    return JSONResponse(content={
        "candidates": store.list_all(),
        "stats":      store.stats(),
    })


@app.get("/api/v1/candidates/{candidate_id}", tags=["Candidates"])
async def get_candidate(candidate_id: str, request: Request, _auth=Depends(require_api_key)):
    """Retrieve a full candidate profile by ID."""
    store: CandidateStore = request.app.state.candidate_store
    record = store.get(candidate_id)
    if not record:
        raise HTTPException(status_code=404, detail={
            "error": "candidate_not_found", "candidate_id": candidate_id
        })
    return JSONResponse(content=record)


@app.get("/api/v1/candidates/{candidate_id}/skills", tags=["Candidates"])
async def get_candidate_skills(candidate_id: str, request: Request, _auth=Depends(require_api_key)):
    """Return just the skill list for a stored candidate."""
    store: CandidateStore = request.app.state.candidate_store
    skills = store.get_skills(candidate_id)
    if skills is None:
        raise HTTPException(status_code=404, detail={
            "error": "candidate_not_found", "candidate_id": candidate_id
        })
    return JSONResponse(content={"candidate_id": candidate_id, "skills": skills, "count": len(skills)})


@app.delete("/api/v1/candidates/{candidate_id}", tags=["Candidates"])
async def delete_candidate(candidate_id: str, request: Request, _auth=Depends(require_api_key)):
    """Delete a candidate record."""
    store: CandidateStore = request.app.state.candidate_store
    existed = store.delete(candidate_id)
    if not existed:
        raise HTTPException(status_code=404, detail={"error": "candidate_not_found"})
    return JSONResponse(content={"deleted": True, "candidate_id": candidate_id})


# ── Skills & Taxonomy ─────────────────────────────────────────────────────────
@app.get("/api/v1/skills/taxonomy", tags=["Skills"])
async def get_skill_taxonomy(request: Request):
    """Browse the full hierarchical skill taxonomy."""
    normalizer: NormalizerAgent = request.app.state.normalizer
    taxonomy = normalizer.get_taxonomy()
    total = sum(
        len(skills)
        for subcats in taxonomy.values()
        for skills in subcats.values()
    )
    return JSONResponse(content={"taxonomy": taxonomy, "total_skills": total})


@app.post("/api/v1/skills/normalize", tags=["Skills"])
async def normalize_skills(payload: dict, request: Request, _auth=Depends(require_api_key)):
    """Normalize a list of raw skill strings against the taxonomy."""
    skills = payload.get("skills", [])
    if not isinstance(skills, list):
        raise HTTPException(status_code=422, detail={"error": "skills must be a list"})
    normalizer: NormalizerAgent = request.app.state.normalizer
    normalized = normalizer.normalize_batch(skills)
    return JSONResponse(content={"normalized": normalized, "count": len(normalized)})


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
