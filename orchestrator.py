"""
Orchestrator: Coordinates Parser, Normalizer, and Matcher agents.
Manages the full analysis pipeline with execution tracing.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from agents.parser_agent import ParserAgent
from agents.normalizer_agent import NormalizerAgent
from agents.matcher_agent import MatcherAgent


class Orchestrator:
    """
    Multi-agent orchestration layer.
    Manages agent lifecycle, task delegation, and result synthesis.
    """

    def __init__(self, parser: ParserAgent, normalizer: NormalizerAgent, matcher: MatcherAgent):
        self.parser = parser
        self.normalizer = normalizer
        self.matcher = matcher
        self._execution_logs: List[Dict] = []

    def _log(self, agent: str, action: str, detail: str = "", duration_ms: float = 0) -> Dict:
        entry = {
            "timestamp": round(time.time() * 1000),
            "agent": agent,
            "action": action,
            "detail": detail,
            "duration_ms": round(duration_ms),
            "status": "success"
        }
        self._execution_logs.append(entry)
        return entry

    async def full_analysis(
        self,
        resume_text: str,
        job_description: str,
        candidate_name: str = "Candidate",
        years_exp: int = 0
    ) -> Dict[str, Any]:
        """
        Full multi-agent pipeline:
        1. ParserAgent → extract raw skills from resume text
        2. NormalizerAgent → normalize + infer skills
        3. MatcherAgent → extract JD skills + compute match
        4. Synthesize → generate insight, roles, learning path
        """
        self._execution_logs = []
        start = time.time()

        # ─── Stage 1: Parse ───────────────────────────────────────────────
        t0 = time.time()
        self._log("Orchestrator", "pipeline_start", f"Analyzing candidate: {candidate_name}")
        
        raw_skills = self.parser._extract_skills_from_text(resume_text)
        years_exp = self.parser._extract_years(resume_text) or years_exp
        
        t1 = time.time()
        self._log("ParserAgent", "skill_extraction", 
                  f"Extracted {len(raw_skills)} raw skills from resume text",
                  (t1 - t0) * 1000)

        # ─── Stage 2: Normalize ───────────────────────────────────────────
        t0 = time.time()
        normalized = self.normalizer.normalize_batch(raw_skills)
        canonical_skills = list({n["canonical"] for n in normalized})
        
        t1 = time.time()
        self._log("NormalizerAgent", "skill_normalization",
                  f"Normalized {len(raw_skills)} → {len(canonical_skills)} canonical skills",
                  (t1 - t0) * 1000)

        # ─── Stage 2b: Inference ──────────────────────────────────────────
        t0 = time.time()
        inferred_skills = self.normalizer.infer_skills(canonical_skills)
        
        t1 = time.time()
        self._log("NormalizerAgent", "skill_inference",
                  f"Inferred {len(inferred_skills)} additional skills from skill graph",
                  (t1 - t0) * 1000)

        # ─── Stage 3: Extract JD Skills ───────────────────────────────────
        t0 = time.time()
        jd_skills = self.matcher.extract_jd_skills(job_description)
        
        t1 = time.time()
        self._log("MatcherAgent", "jd_parsing",
                  f"Extracted {len(jd_skills['required'])} required + {len(jd_skills['nice_to_have'])} nice-to-have skills from JD",
                  (t1 - t0) * 1000)

        # ─── Stage 4: Semantic Match ──────────────────────────────────────
        t0 = time.time()
        match_result = self.matcher.compute_match(
            canonical_skills,
            jd_skills["required"],
            jd_skills["nice_to_have"],
            inferred_skills
        )
        
        t1 = time.time()
        method = match_result.get("method", "unknown")
        self._log("MatcherAgent", "semantic_matching",
                  f"Computed match score: {match_result['match_score']}/100 using {method}",
                  (t1 - t0) * 1000)

        # ─── Stage 5: Role Suggestions ────────────────────────────────────
        t0 = time.time()
        suggested_roles = self.normalizer.suggest_roles(canonical_skills + inferred_skills)
        t1 = time.time()
        self._log("NormalizerAgent", "role_suggestion",
                  f"Suggested {len(suggested_roles)} matching roles",
                  (t1 - t0) * 1000)

        # ─── Stage 6: Learning Recommendations ───────────────────────────
        t0 = time.time()
        missing_skill_names = [m["skill"] for m in match_result["missing_skills"][:5]]
        learning_recs = self.normalizer.get_learning_recommendations(missing_skill_names)
        t1 = time.time()
        self._log("NormalizerAgent", "learning_path",
                  f"Generated {len(learning_recs)} learning recommendations",
                  (t1 - t0) * 1000)

        # ─── Stage 7: AI Insight ──────────────────────────────────────────
        t0 = time.time()
        ai_insight = self.matcher.generate_ai_insight(
            candidate_name,
            match_result["match_score"],
            match_result["matched_skills"],
            match_result["missing_skills"],
            inferred_skills,
            years_exp
        )
        t1 = time.time()
        self._log("Orchestrator", "insight_generation",
                  "Generated explainable AI assessment",
                  (t1 - t0) * 1000)

        # ─── Finalize ────────────────────────────────────────────────────
        total_ms = round((time.time() - start) * 1000)
        self._log("Orchestrator", "pipeline_complete",
                  f"Full analysis completed in {total_ms}ms", total_ms)

        return {
            "candidate_name": candidate_name,
            "match_score": match_result["match_score"],
            "confidence": match_result["confidence"],
            "years_of_experience": years_exp,
            "matched_skills": match_result["matched_skills"],
            "missing_skills": match_result["missing_skills"],
            "all_candidate_skills": canonical_skills,
            "inferred_skills": inferred_skills,
            "suggested_roles": suggested_roles,
            "learning_recommendations": learning_recs,
            "ai_insight": ai_insight,
            "jd_required_skills": jd_skills["required"],
            "jd_nice_skills": jd_skills["nice_to_have"],
            "agent_logs": self._execution_logs,
            "match_method": method,
            "status": "success",
        }
