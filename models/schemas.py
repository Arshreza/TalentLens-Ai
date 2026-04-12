"""
TalentLens AI — Pydantic Schemas (v2.1)
Covers all API request/response models.
"""

from pydantic import BaseModel, field_validator, HttpUrl
from typing import Optional, List, Dict, Any


# ── Requests ──────────────────────────────────────────────────────────────────

class MatchRequest(BaseModel):
    resume_text: str
    job_description: str
    candidate_name: Optional[str] = "Candidate"
    expected_years: Optional[int] = 0
    webhook_url: Optional[str] = None   # POST result here when done (async)

    @field_validator("resume_text")
    @classmethod
    def resume_not_empty(cls, v):
        if not v or len(v.strip()) < 50:
            raise ValueError("Resume text must be at least 50 characters.")
        return v.strip()

    @field_validator("job_description")
    @classmethod
    def jd_not_empty(cls, v):
        if not v or len(v.strip()) < 30:
            raise ValueError("Job description must be at least 30 characters.")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "resume_text": "Priya Mehta — ML Engineer with 4 years experience. Python, PyTorch, FastAPI, Docker, Kubernetes, LangChain.",
                "job_description": "Senior ML Engineer. Required: Python, PyTorch, Docker, Kubernetes, FastAPI. Nice to have: AWS, Kafka.",
                "candidate_name": "Priya Mehta",
                "expected_years": 4,
                "webhook_url": None,
            }
        }


class ParseRequest(BaseModel):
    text: str


class NormalizeRequest(BaseModel):
    skills: List[str]


# ── Sub-models ────────────────────────────────────────────────────────────────

class SkillMatch(BaseModel):
    skill: str
    jd_skill: str
    match_type: str          # exact | inferred | semantic
    similarity: Optional[float] = None


class GapItem(BaseModel):
    skill: str
    importance: str          # required | nice_to_have


class AgentLogEntry(BaseModel):
    agent: str
    action: str
    detail: str
    start_time: float
    end_time: float
    duration_ms: float
    status: str
    stage: Optional[int] = None


class MatchBreakdown(BaseModel):
    exact: int
    inferred: int
    semantic: int
    total_matched: int
    total_required: int


class EducationEntry(BaseModel):
    degree: str
    line: str


class ExperienceEntry(BaseModel):
    role: Optional[str] = ""
    company: Optional[str] = ""
    duration: Optional[str] = ""
    responsibilities: Optional[List[str]] = []


# ── Responses ─────────────────────────────────────────────────────────────────

class ParsedResume(BaseModel):
    name: str
    email: str
    phone: str
    linkedin: str
    github: str
    summary: str
    raw_skills: List[str]
    years_of_experience: int
    education: List[Dict[str, Any]]
    experience: List[Dict[str, Any]]
    certifications: List[str]
    projects: List[str]
    location: str
    sections_detected: List[str]
    raw_text_length: int
    filename: str
    status: str


class AnalysisResult(BaseModel):
    candidate_id: Optional[str] = None
    candidate_name: str
    match_score: int
    confidence: float
    hiring_recommendation: str
    years_of_experience: int
    matched_skills: List[Dict[str, Any]]
    missing_skills: List[Dict[str, Any]]
    all_candidate_skills: List[str]
    inferred_skills: List[str]
    suggested_roles: List[str]
    learning_recommendations: List[str]
    ai_insight: str
    jd_required_skills: List[str]
    jd_nice_skills: List[str]
    match_breakdown: Dict[str, int]
    agent_logs: List[Dict[str, Any]]
    match_method: str
    total_pipeline_ms: int
    api_processing_ms: Optional[int] = None
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str          # pending | running | complete | failed
    created_at: float
    completed_at: Optional[float] = None
    description: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class CandidateSummary(BaseModel):
    candidate_id: str
    candidate_name: str
    match_score: int
    hiring_recommendation: str
    stored_at: float
    source_file: str
    skill_count: int


class BatchParseResult(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: List[Dict[str, Any]]
    processing_time_ms: int
