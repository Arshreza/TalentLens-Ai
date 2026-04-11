from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class MatchRequest(BaseModel):
    resume_text: str
    job_description: str
    candidate_name: Optional[str] = "Candidate"

    class Config:
        json_schema_extra = {
            "example": {
                "resume_text": "John Doe. Python developer with 5 years of experience in ML, TensorFlow, FastAPI...",
                "job_description": "We need a Senior ML Engineer with Python, PyTorch, Docker, Kubernetes...",
                "candidate_name": "John Doe"
            }
        }


class ParseRequest(BaseModel):
    text: str


class SkillMatch(BaseModel):
    skill: str
    canonical: str
    confidence: float
    category: str


class GapItem(BaseModel):
    skill: str
    importance: str  # required / nice-to-have
    suggestion: str  # how to acquire


class AnalysisResult(BaseModel):
    candidate_name: str
    match_score: int
    confidence: float
    matched_skills: List[SkillMatch]
    missing_skills: List[GapItem]
    inferred_skills: List[str]
    suggested_roles: List[str]
    learning_recommendations: List[str]
    ai_insight: str
    agent_logs: List[Dict[str, Any]]
    processing_time_ms: int
