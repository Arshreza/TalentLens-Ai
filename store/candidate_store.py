"""
In-memory Candidate Store for TalentLens AI.
Stores analysis results keyed by UUID candidate IDs.
Supports TTL-based eviction (default 24 hours).
For production: swap with PostgreSQL + SQLAlchemy.
"""

import time
import uuid
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("talentlens.store")

# TTL in seconds (default: 24 hours)
DEFAULT_TTL = 86400


class CandidateStore:
    """
    Thread-safe in-memory candidate profile store.
    Each entry stores the full analysis result + metadata.

    Production migration path:
        Replace _store dict operations with SQLAlchemy session calls.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds

    def add(self, analysis_result: Dict[str, Any], source_file: str = "") -> str:
        """Store a new candidate profile. Returns the generated candidate_id."""
        candidate_id = str(uuid.uuid4())
        self._store[candidate_id] = {
            "candidate_id": candidate_id,
            "stored_at": time.time(),
            "source_file": source_file,
            "analysis": analysis_result,
            # Quick-access summary fields
            "candidate_name": analysis_result.get("candidate_name", "Unknown"),
            "match_score": analysis_result.get("match_score", 0),
            "hiring_recommendation": analysis_result.get("hiring_recommendation", "N/A"),
            "all_candidate_skills": analysis_result.get("all_candidate_skills", []),
        }
        logger.info(f"[Store] Saved candidate {candidate_id} — {analysis_result.get('candidate_name')}")
        self._evict_expired()
        return candidate_id

    def get(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a candidate by ID. Returns None if not found or expired."""
        record = self._store.get(candidate_id)
        if not record:
            return None
        if time.time() - record["stored_at"] > self._ttl:
            del self._store[candidate_id]
            return None
        return record

    def list_all(self) -> List[Dict[str, Any]]:
        """Return summary list of all stored candidates (no full analysis)."""
        self._evict_expired()
        summaries = []
        for cid, rec in self._store.items():
            summaries.append({
                "candidate_id": cid,
                "candidate_name": rec["candidate_name"],
                "match_score": rec["match_score"],
                "hiring_recommendation": rec["hiring_recommendation"],
                "stored_at": rec["stored_at"],
                "source_file": rec["source_file"],
                "skill_count": len(rec["all_candidate_skills"]),
            })
        summaries.sort(key=lambda x: x["stored_at"], reverse=True)
        return summaries

    def get_skills(self, candidate_id: str) -> Optional[List[str]]:
        """Return just the skills list for a candidate."""
        record = self.get(candidate_id)
        if not record:
            return None
        return record.get("all_candidate_skills", [])

    def delete(self, candidate_id: str) -> bool:
        """Delete a candidate record. Returns True if it existed."""
        existed = candidate_id in self._store
        self._store.pop(candidate_id, None)
        return existed

    def stats(self) -> Dict[str, Any]:
        self._evict_expired()
        scores = [r["match_score"] for r in self._store.values()]
        return {
            "total_candidates": len(self._store),
            "avg_match_score": round(sum(scores) / max(1, len(scores)), 1),
            "strong_hires": sum(1 for r in self._store.values() if r["hiring_recommendation"] == "Strong Hire"),
        }

    def _evict_expired(self):
        """Remove entries older than TTL."""
        now = time.time()
        expired = [k for k, v in self._store.items() if now - v["stored_at"] > self._ttl]
        for k in expired:
            del self._store[k]
        if expired:
            logger.debug(f"[Store] Evicted {len(expired)} expired entries.")


# ── Job Queue (async background jobs) ────────────────────────────────────────

class JobQueue:
    """
    Simple in-memory async job tracker for background analysis tasks.
    Production: swap with Redis/Celery task IDs.
    """

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create(self, description: str = "") -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": time.time(),
            "completed_at": None,
            "description": description,
            "result": None,
            "error": None,
        }
        return job_id

    def set_running(self, job_id: str):
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = "running"

    def set_complete(self, job_id: str, result: Any):
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = "complete"
            self._jobs[job_id]["completed_at"] = time.time()
            self._jobs[job_id]["result"] = result

    def set_failed(self, job_id: str, error: str):
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = "failed"
            self._jobs[job_id]["completed_at"] = time.time()
            self._jobs[job_id]["error"] = error

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)
