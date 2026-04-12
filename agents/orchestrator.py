"""
Orchestrator v3.0: Coordinates Parser, Normalizer, and Matcher agents.
Manages the full analysis pipeline with async job support and webhook callbacks.
"""

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional

from agents.parser_agent import ParserAgent
from agents.normalizer_agent import NormalizerAgent
from agents.matcher_agent import MatcherAgent

logger = logging.getLogger("talentlens.orchestrator")


class Orchestrator:
    """
    Multi-agent orchestration layer.
    Manages agent lifecycle, task delegation, stage tracking, and result synthesis.
    Supports sync (full_analysis) and async (async_analyze) execution modes.
    """

    def __init__(self, parser: ParserAgent, normalizer: NormalizerAgent, matcher: MatcherAgent):
        self.parser     = parser
        self.normalizer = normalizer
        self.matcher    = matcher
        self._execution_logs: List[Dict] = []

    # ── Internal Logging ───────────────────────────────────────────────────────

    def _log(self, agent: str, action: str, detail: str = "",
             start_time: float = 0, duration_ms: float = 0, stage: int = 0) -> Dict:
        if not start_time:
            start_time = time.time()
        entry = {
            "start_time":  start_time,
            "end_time":    start_time + (duration_ms / 1000.0),
            "agent":       agent,
            "action":      action,
            "detail":      detail,
            "duration_ms": round(duration_ms, 2),
            "status":      "success",
            "stage":       stage,
        }
        self._execution_logs.append(entry)
        logger.debug(f"[Stage {stage}] {agent} — {action}: {detail}")
        return entry

    # ── Full Synchronous Pipeline ──────────────────────────────────────────────

    async def full_analysis(
        self,
        resume_text: str,
        job_description: str,
        candidate_name: str = "Candidate",
        years_exp: int = 0,
    ) -> Dict[str, Any]:
        """
        7-stage multi-agent pipeline.
        Returns the complete analysis result dict.
        """
        self._execution_logs = []
        pipeline_start = time.time()

        # Stage 0 — Pipeline Start
        self._log("Orchestrator", "pipeline_start",
                  f"Initializing 7-stage pipeline for '{candidate_name}'",
                  start_time=pipeline_start, duration_ms=1, stage=0)
        await asyncio.sleep(0)

        # Stage 1 — Parse Resume
        t0 = time.time()
        raw_skills = self.parser._extract_skills_from_text(resume_text)
        experience_years = self.parser._extract_years(resume_text) or years_exp
        t1 = time.time()
        self._log("ParserAgent", "skill_extraction",
                  f"Extracted {len(raw_skills)} raw skill tokens from resume text",
                  start_time=t0, duration_ms=(t1 - t0) * 1000, stage=1)
        await asyncio.sleep(0)

        # Stage 2 — Normalize Skills
        t0 = time.time()
        normalized = self.normalizer.normalize_batch(raw_skills)
        canonical_skills = list({n["canonical"] for n in normalized})
        t1 = time.time()
        self._log("NormalizerAgent", "skill_normalization",
                  f"Normalized → {len(canonical_skills)} canonical skills via taxonomy lookup",
                  start_time=t0, duration_ms=(t1 - t0) * 1000, stage=2)
        await asyncio.sleep(0)

        # Stage 2b — Skill Inference
        t0 = time.time()
        inferred_skills = self.normalizer.infer_skills(canonical_skills)
        t1 = time.time()
        self._log("NormalizerAgent", "skill_inference",
                  f"Inferred {len(inferred_skills)} additional skills from the skill graph",
                  start_time=t0, duration_ms=(t1 - t0) * 1000, stage=2)
        await asyncio.sleep(0)

        # Stage 3 — Parse Job Description
        t0 = time.time()
        jd_skills = self.matcher.extract_jd_skills(job_description)
        t1 = time.time()
        self._log("MatcherAgent", "jd_parsing",
                  f"Identified {len(jd_skills['required'])} required + "
                  f"{len(jd_skills['nice_to_have'])} nice-to-have skills from JD",
                  start_time=t0, duration_ms=(t1 - t0) * 1000, stage=3)
        await asyncio.sleep(0)

        # Stage 4 — Semantic Match (offloaded to thread for SBERT)
        t0 = time.time()
        match_result = await asyncio.to_thread(
            self.matcher.compute_match,
            canonical_skills,
            jd_skills["required"],
            jd_skills["nice_to_have"],
            inferred_skills,
        )
        t1 = time.time()
        bd = match_result.get("match_breakdown", {})
        method = match_result.get("method", "unknown")
        self._log("MatcherAgent", "semantic_matching",
                  f"Match score: {match_result['match_score']}/100 — "
                  f"Exact: {bd.get('exact',0)}, Inferred: {bd.get('inferred',0)}, "
                  f"Semantic: {bd.get('semantic',0)} — via {method}",
                  start_time=t0, duration_ms=(t1 - t0) * 1000, stage=4)
        await asyncio.sleep(0)

        # Stage 5 — Role Suggestions
        t0 = time.time()
        suggested_roles = self.normalizer.suggest_roles(canonical_skills + inferred_skills)
        t1 = time.time()
        self._log("NormalizerAgent", "role_suggestion",
                  f"Matched {len(suggested_roles)} role(s) from candidate skill profile",
                  start_time=t0, duration_ms=(t1 - t0) * 1000, stage=5)
        await asyncio.sleep(0)

        # Stage 6 — Learning Path
        t0 = time.time()
        missing_names = [m["skill"] for m in match_result["missing_skills"][:5]]
        learning_recs = self.normalizer.get_learning_recommendations(missing_names)
        t1 = time.time()
        self._log("NormalizerAgent", "learning_path",
                  f"Generated {len(learning_recs)} targeted learning recommendation(s)",
                  start_time=t0, duration_ms=(t1 - t0) * 1000, stage=6)
        await asyncio.sleep(0)

        # Stage 7 — AI Insight + Hiring Recommendation
        t0 = time.time()
        req_missing_count = sum(1 for m in match_result["missing_skills"] if m.get("importance") == "required")
        hiring_rec = self.matcher.get_hiring_recommendation(
            match_result["match_score"], req_missing_count, len(jd_skills["required"])
        )
        ai_insight = self.matcher.generate_ai_insight(
            candidate_name,
            match_result["match_score"],
            match_result["matched_skills"],
            match_result["missing_skills"],
            inferred_skills,
            experience_years,
        )
        t1 = time.time()
        self._log("Orchestrator", "insight_generation",
                  f"Generated explainable AI assessment — Hiring recommendation: {hiring_rec}",
                  start_time=t0, duration_ms=(t1 - t0) * 1000, stage=7)
        await asyncio.sleep(0)

        # Synthesis
        pipeline_end = time.time()
        total_ms = round((pipeline_end - pipeline_start) * 1000)
        self._log("Orchestrator", "pipeline_complete",
                  f"Full pipeline completed in {total_ms}ms",
                  start_time=pipeline_end, duration_ms=1, stage=7)

        return {
            "candidate_name":        candidate_name,
            "match_score":           match_result["match_score"],
            "confidence":            match_result["confidence"],
            "hiring_recommendation": hiring_rec,
            "years_of_experience":   experience_years,
            "matched_skills":        match_result["matched_skills"],
            "missing_skills":        match_result["missing_skills"],
            "all_candidate_skills":  canonical_skills,
            "inferred_skills":       inferred_skills,
            "suggested_roles":       suggested_roles,
            "learning_recommendations": learning_recs,
            "ai_insight":            ai_insight,
            "jd_required_skills":    jd_skills["required"],
            "jd_nice_skills":        jd_skills["nice_to_have"],
            "match_breakdown":       match_result.get("match_breakdown", {}),
            "agent_logs":            self._execution_logs,
            "match_method":          method,
            "total_pipeline_ms":     total_ms,
            "status":                "success",
        }

    # ── Async Background Job ───────────────────────────────────────────────────

    async def async_analyze(
        self,
        resume_text: str,
        job_description: str,
        candidate_name: str,
        job_queue,          # store.candidate_store.JobQueue
        candidate_store,    # store.candidate_store.CandidateStore
        job_id: str,
        webhook_url: Optional[str] = None,
    ):
        """
        Run analysis in the background. Updates job_queue status and stores result.
        Optionally fires a webhook POST when complete.
        """
        job_queue.set_running(job_id)
        retries = 0
        max_retries = 2
        last_error = None

        while retries <= max_retries:
            try:
                result = await self.full_analysis(resume_text, job_description, candidate_name)
                candidate_id = candidate_store.add(result)
                result["candidate_id"] = candidate_id
                job_queue.set_complete(job_id, result)

                if webhook_url:
                    await self._fire_webhook(webhook_url, job_id, result)

                logger.info(f"[Job {job_id}] Completed successfully — candidate_id={candidate_id}")
                return
            except Exception as e:
                retries += 1
                last_error = str(e)
                logger.warning(f"[Job {job_id}] Attempt {retries} failed: {e}")
                if retries <= max_retries:
                    await asyncio.sleep(1)

        job_queue.set_failed(job_id, last_error)
        logger.error(f"[Job {job_id}] All retries exhausted. Error: {last_error}")

    async def _fire_webhook(self, url: str, job_id: str, result: Dict[str, Any]):
        """Fire an async HTTP POST webhook with the analysis result."""
        try:
            import httpx
            payload = {"job_id": job_id, "status": "complete", "result": result}
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(url, json=payload)
                logger.info(f"[Webhook] POST {url} → {resp.status_code}")
        except Exception as e:
            logger.warning(f"[Webhook] Failed to deliver to {url}: {e}")
