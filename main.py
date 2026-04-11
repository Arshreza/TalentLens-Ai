"""
TalentLens AI - Multi-Agent Resume Intelligence Backend
FastAPI + Sentence Transformers + Multi-Agent Orchestration
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import time
import asyncio

from agents.orchestrator import Orchestrator
from agents.parser_agent import ParserAgent
from agents.normalizer_agent import NormalizerAgent
from agents.matcher_agent import MatcherAgent
from models.schemas import ParseRequest, MatchRequest, AnalysisResult

app = FastAPI(
    title="TalentLens AI",
    description="Multi-Agent Resume Intelligence API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
parser_agent = ParserAgent()
normalizer_agent = NormalizerAgent()
matcher_agent = MatcherAgent()
orchestrator = Orchestrator(parser_agent, normalizer_agent, matcher_agent)


@app.get("/")
async def root():
    return {"status": "TalentLens AI is running", "version": "1.0.0", "agents": ["Parser", "Normalizer", "Matcher", "Orchestrator"]}


@app.post("/api/v1/parse")
async def parse_resume(file: UploadFile = File(...)):
    """
    Parse a resume file (PDF or DOCX) and extract structured information.
    Returns: name, email, skills, experience, education, summary
    """
    start = time.time()
    
    if not file.filename.endswith(('.pdf', '.docx', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, or TXT files are supported")
    
    content = await file.read()
    
    try:
        result = await parser_agent.parse(content, file.filename)
        result["processing_time_ms"] = round((time.time() - start) * 1000)
        result["agent"] = "ParserAgent v1.0"
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {str(e)}")


@app.post("/api/v1/match")
async def match_candidate(request: MatchRequest):
    """
    Match a candidate profile against a job description.
    Returns: match_score, matched_skills, missing_skills, gap_analysis, ai_insight
    """
    start = time.time()
    
    try:
        result = await orchestrator.full_analysis(
            resume_text=request.resume_text,
            job_description=request.job_description,
            candidate_name=request.candidate_name
        )
        result["processing_time_ms"] = round((time.time() - start) * 1000)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Match error: {str(e)}")


@app.get("/api/v1/skills/taxonomy")
async def get_skill_taxonomy():
    """
    Browse the skill taxonomy hierarchy.
    Returns: hierarchical skill categories with examples
    """
    return JSONResponse(content=normalizer_agent.get_taxonomy())


@app.post("/api/v1/skills/normalize")
async def normalize_skills(skills: list[str]):
    """
    Normalize a list of raw skill strings against the taxonomy.
    """
    normalized = normalizer_agent.normalize_batch(skills)
    return JSONResponse(content={"normalized": normalized, "count": len(normalized)})


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "agents_loaded": True}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
