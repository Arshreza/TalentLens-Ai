"""
main.py — FastAPI backend for TalentLens AI Resume Intelligence Platform.
Includes:
  - /api/v1/parse   → PDF / DOCX / TXT text extraction
  - /api/v1/match   → Full resume analysis with skills, score, roles, insight
  - /api/v1/health  → Health check
"""

import io
import time
import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# model.py semantic scorer is no longer used in the primary scoring pipeline
from skills import (
    extract_skills,
    infer_skills,
    compare_skills,
    suggest_roles,
)

app = FastAPI(
    title="TalentLens AI",
    description="Multi-agent resume intelligence platform",
    version="2.0.0",
)

# Allow frontend (any origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def make_log(agent: str, action: str, detail: str, duration_ms: int = 0) -> dict:
    return {
        "timestamp": now_iso(),
        "agent": agent,
        "action": action,
        "detail": detail,
        "duration_ms": duration_ms,
    }

from utils import clean_text

def extract_text_from_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data), strict=False)
        pages = [page.extract_text() or "" for page in reader.pages]
        return clean_text("\n".join(pages))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF parsing failed: {e}")

def extract_text_from_docx(data: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(data))
        return clean_text("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"DOCX parsing failed: {e}")

def build_ai_insight(
    candidate_name: str,
    score: int,
    matched: list,
    missing: list,
    inferred: list,
    roles: list,
) -> str:
    """Generate a human-readable AI insight paragraph."""
    strength_str = ", ".join(matched[:4]) if matched else "no direct skill matches"
    gap_str     = ", ".join(missing[:3]) if missing else "none identified"
    infer_str   = ", ".join(inferred[:3]) if inferred else ""
    role_str    = ", ".join(roles[:2]) if roles else "General Engineering"

    if score >= 80:
        verdict = "highly recommended for interview"
        level   = "Strong"
    elif score >= 60:
        verdict = "a solid candidate worth interviewing"
        level   = "Moderate"
    elif score >= 40:
        verdict = "a potential candidate with notable skill gaps"
        level   = "Partial"
    else:
        verdict = "not a strong match at this time"
        level   = "Weak"

    insight = (
        f"{candidate_name} shows a {level.lower()} match ({score}/100) for this role. "
        f"Core strengths include: {strength_str}. "
    )
    if infer_str:
        insight += f"Additionally, proficiency in {infer_str} is inferred from their skill set. "
    if missing:
        insight += f"Key gaps to address: {gap_str}. "
    insight += (
        f"Best-fit roles based on profile: {role_str}. "
        f"Overall assessment: {verdict}."
    )
    return insight

def generate_learning_recs(missing_skills: list) -> list[str]:
    """Generate actionable learning recommendations for missing skills."""
    RESOURCES = {
        "Kubernetes":    "KCNA certification → kubernetes.io/docs",
        "LLMs":          "Hugging Face NLP Course (free) → huggingface.co/learn",
        "AWS":           "AWS Cloud Practitioner → aws.amazon.com/training",
        "Docker":        "Docker Getting Started → docs.docker.com",
        "React":         "Official React Docs → react.dev",
        "TypeScript":    "TypeScript Handbook → typescriptlang.org/docs",
        "Terraform":     "HashiCorp Learn → developer.hashicorp.com/terraform",
        "Kafka":         "Confluent Kafka 101 → developer.confluent.io",
        "Spark":         "Databricks Academy (free) → academy.databricks.com",
        "PostgreSQL":    "PostgreSQL Tutorial → postgresqltutorial.com",
        "Machine Learning": "fast.ai Practical ML → fast.ai",
        "Deep Learning": "DeepLearning.AI specialization → deeplearning.ai",
        "NLP":           "Stanford CS224N → web.stanford.edu/class/cs224n",
        "CI/CD":         "GitHub Actions Docs → docs.github.com/actions",
        "System Design": "System Design Primer → github.com/donnemartin",
        "Go":            "Go Tour → tour.golang.org",
        "Rust":          "The Rust Book → doc.rust-lang.org/book",
    }
    recs = []
    for skill in missing_skills[:4]:
        hint = RESOURCES.get(skill, f"Search: '{skill} tutorial' on Coursera or Udemy")
        recs.append(f"Learn {skill}: {hint}")
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/api/v1/parse")
async def parse_resume(file: UploadFile = File(...)):
    """
    Accept PDF, DOCX, or TXT file and return extracted plain text.
    """
    t0 = time.time()
    data = await file.read()
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        raw_text = extract_text_from_pdf(data)
    elif ext in ("docx", "doc"):
        raw_text = extract_text_from_docx(data)
    elif ext == "txt":
        raw_text = data.decode("utf-8", errors="replace")
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: .{ext}")

    duration = int((time.time() - t0) * 1000)
    return {
        "filename": filename,
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "parse_time_ms": duration,
        "status": "success",
    }


class MatchRequest(BaseModel):
    resume_text: str
    job_description: str
    candidate_name: str = "Candidate"


@app.post("/api/v1/match")
def match(req: MatchRequest):
    """
    Full resume-vs-JD analysis pipeline.
    Returns score, skills breakdown, inferred skills, roles, insight, logs.
    """
    logs = []
    pipeline_start = time.time()

    # ── Step 1: Extract skills from resume ──────────────────────────────────
    t = time.time()
    resume_skills = extract_skills(req.resume_text)
    logs.append(make_log(
        "ParserAgent", "skill_extraction",
        f"Extracted {len(resume_skills)} canonical skills from resume",
        int((time.time() - t) * 1000)
    ))

    # ── Step 2: Infer additional skills ─────────────────────────────────────
    t = time.time()
    inferred = infer_skills(resume_skills)
    all_resume_skills = resume_skills + [s for s in inferred if s not in resume_skills]
    logs.append(make_log(
        "NormalizerAgent", "skill_inference",
        f"Inferred {len(inferred)} additional skills from skill graph",
        int((time.time() - t) * 1000)
    ))

    # ── Step 3: Extract skills from JD ──────────────────────────────────────
    t = time.time()
    jd_skills = extract_skills(req.job_description)
    logs.append(make_log(
        "MatcherAgent", "jd_parsing",
        f"Extracted {len(jd_skills)} required skills from job description",
        int((time.time() - t) * 1000)
    ))

    # ── Step 4: Compare skills ───────────────────────────────────────────────
    t = time.time()
    matched, missing = compare_skills(all_resume_skills, jd_skills)
    logs.append(make_log(
        "MatcherAgent", "skill_matching",
        f"Matched {len(matched)} skills, {len(missing)} missing",
        int((time.time() - t) * 1000)
    ))

    # ── Step 5: Skill-based score ─────────────────────────────────────────────
    # Score = how many of the JD's required skills the candidate has.
    # Extra/redundant skills in resume do NOT reduce the score.
    if jd_skills:
        skill_overlap_ratio = len(matched) / max(len(jd_skills), 1)
        score = int(round(skill_overlap_ratio * 100))
    else:
        # No skills detected in JD — give benefit of the doubt
        score = 50
    confidence = round(0.5 + abs((score / 100) - 0.5) * 0.6, 4)
    logs.append(make_log(
        "MatcherAgent", "skill_overlap_scoring",
        f"Computed skill-match score: {score}/100 ({len(matched)}/{len(jd_skills)} required skills matched)",
        0
    ))

    # ── Step 7: Suggest roles ────────────────────────────────────────────────
    t = time.time()
    roles = suggest_roles(all_resume_skills)
    logs.append(make_log(
        "NormalizerAgent", "role_suggestion",
        f"Suggested {len(roles)} matching roles based on skill profile",
        int((time.time() - t) * 1000)
    ))

    # ── Step 8: Learning recommendations ────────────────────────────────────
    t = time.time()
    learning_recs = generate_learning_recs(missing)
    logs.append(make_log(
        "NormalizerAgent", "learning_path",
        f"Generated {len(learning_recs)} learning recommendations",
        int((time.time() - t) * 1000)
    ))

    # ── Step 9: AI Insight ───────────────────────────────────────────────────
    t = time.time()
    ai_insight = build_ai_insight(
        req.candidate_name, score,
        matched, missing, inferred, roles
    )
    logs.append(make_log(
        "Orchestrator", "insight_generation",
        "Generated explainable AI assessment",
        int((time.time() - t) * 1000)
    ))

    total_ms = int((time.time() - pipeline_start) * 1000)
    logs.append(make_log(
        "Orchestrator", "pipeline_complete",
        f"Full analysis completed in {total_ms}ms",
        total_ms
    ))

    # ── Build insight text ───────────────────────────────────────────────────
    if score > 75:
        short_insight = f"Strong match ({score}/100). Candidate fits well."
    elif score > 50:
        short_insight = f"Moderate match ({score}/100). Some skill gaps present."
    elif score > 30:
        short_insight = f"Partial match ({score}/100). Notable skill gaps."
    else:
        short_insight = f"Weak match ({score}/100). Major skills are missing."

    return {
        "candidate_name": req.candidate_name,
        "match_score":    score,
        "confidence":     round(confidence, 4),

        # Skills breakdown
        "matched_skills":  [{"jd_skill": s, "importance": "required"} for s in matched],
        "missing_skills":  [{"skill": s, "importance": "required"} for s in missing],
        "inferred_skills": inferred,

        # Roles & learning
        "suggested_roles":          roles,
        "learning_recommendations": learning_recs,

        # Insights
        "ai_insight":       ai_insight,
        "short_insight":    short_insight,

        # Metadata
        "agent_logs":       logs,
        "resume_skill_count": len(resume_skills),
        "jd_skill_count":     len(jd_skills),
        "match_method":       "weighted_semantic_+_skill_overlap",
        "status":             "success",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)