"""
MatcherAgent: Computes semantic similarity between candidate skills and job description.
Uses Sentence-Transformers (SBERT) when available, falls back to exact/heuristic matching.
Returns hiring recommendations, match breakdowns, and recruiter-quality AI insights.
"""

import re
from typing import List, Dict, Any
import numpy as np
from agents.knowledge_base import KNOWN_SKILLS

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


class MatcherAgent:
    """
    Semantic skill matcher with explainable hiring insights.
    Supports exact, inferred, and SBERT-based semantic matching.
    """

    def __init__(self):
        self.model = None
        if HAS_SBERT:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"[MatcherAgent] SBERT load failed: {e}. Using exact matching.")
                global HAS_SBERT
                HAS_SBERT = False

    # ── JD Parsing ─────────────────────────────────────────────────────────

    def extract_jd_skills(self, job_description: str) -> Dict[str, List[str]]:
        """Extract required and nice-to-have skills from JD text."""
        text_lower = job_description.lower()
        found = set()

        for skill in KNOWN_SKILLS:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found.add(skill)

        # Split required vs nice-to-have
        split_pattern = r'(?i)\b(?:nice to have|preferred|bonus|plus|prefer to have|advantage|optionally)\b'
        parts = re.split(split_pattern, job_description)

        req_text = parts[0].lower()
        nice_text = " ".join(parts[1:]).lower() if len(parts) > 1 else ""

        required = []
        nice_to_have = []

        for skill in found:
            skill_lower = skill.lower()
            if skill_lower in nice_text and skill_lower not in req_text:
                nice_to_have.append(skill)
            else:
                required.append(skill)

        return {"required": required, "nice_to_have": nice_to_have}

    # ── Core Match Engine ───────────────────────────────────────────────────

    def compute_match(
        self,
        candidate_skills: List[str],
        jd_required: List[str],
        jd_nice: List[str],
        inferred_skills: List[str]
    ) -> Dict[str, Any]:
        """
        Compute the candidate ↔ JD match score.
        Scores: exact=2.0, inferred=1.6, semantic=1.8×similarity (required)
                exact=1.0, inferred=0.8, semantic=0.9×similarity (nice)
        """
        all_candidate = {s.lower() for s in candidate_skills}
        inferred_set = {s.lower() for s in inferred_skills}

        matched_skills = []
        missing_skills = []
        score_points = 0.0
        total_points = 0.0

        # Breakdown counters
        exact_count = inferred_count = semantic_count = 0

        method = "semantic (SBERT)" if HAS_SBERT and self.model else "exact match"

        # Pre-compute candidate embeddings once
        candidate_embeddings = None
        combined_candidate = list(set(candidate_skills + inferred_skills))
        if HAS_SBERT and self.model and combined_candidate:
            try:
                candidate_embeddings = self.model.encode(combined_candidate)
            except Exception:
                pass

        def _semantic_best(skill: str):
            if not (HAS_SBERT and candidate_embeddings is not None and len(candidate_embeddings) > 0):
                return None, 0.0
            try:
                skill_emb = self.model.encode([skill])
                sims = cosine_similarity(skill_emb, candidate_embeddings)[0]
                idx = int(np.argmax(sims))
                return combined_candidate[idx], float(sims[idx])
            except Exception:
                return None, 0.0

        # Required skills
        for skill in jd_required:
            total_points += 2.0
            skill_l = skill.lower()
            matched = False

            if skill_l in all_candidate:
                matched_skills.append({"skill": skill, "jd_skill": skill, "match_type": "exact"})
                score_points += 2.0
                exact_count += 1
                matched = True

            elif skill_l in inferred_set:
                matched_skills.append({"skill": skill, "jd_skill": skill, "match_type": "inferred"})
                score_points += 1.6
                inferred_count += 1
                matched = True

            else:
                best_skill, best_sim = _semantic_best(skill)
                if best_sim > 0.70:
                    matched_skills.append({
                        "skill": best_skill, "jd_skill": skill,
                        "match_type": "semantic", "similarity": round(best_sim, 3)
                    })
                    score_points += 1.8 * best_sim
                    semantic_count += 1
                    matched = True

            if not matched:
                missing_skills.append({"skill": skill, "importance": "required"})

        # Nice-to-have skills
        for skill in jd_nice:
            total_points += 1.0
            skill_l = skill.lower()
            matched = False

            if skill_l in all_candidate:
                matched_skills.append({"skill": skill, "jd_skill": skill, "match_type": "exact"})
                score_points += 1.0
                exact_count += 1
                matched = True

            elif skill_l in inferred_set:
                matched_skills.append({"skill": skill, "jd_skill": skill, "match_type": "inferred"})
                score_points += 0.8
                inferred_count += 1
                matched = True

            else:
                best_skill, best_sim = _semantic_best(skill)
                if best_sim > 0.70:
                    matched_skills.append({
                        "skill": best_skill, "jd_skill": skill,
                        "match_type": "semantic", "similarity": round(best_sim, 3)
                    })
                    score_points += 0.9 * best_sim
                    semantic_count += 1
                    matched = True

            if not matched:
                missing_skills.append({"skill": skill, "importance": "nice_to_have"})

        final_score = int((score_points / max(1.0, total_points)) * 100) if total_points > 0 else 0
        final_score = min(100, max(0, final_score))
        confidence = 0.88 if HAS_SBERT else 0.62

        return {
            "match_score": final_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "confidence": confidence,
            "method": method,
            "match_breakdown": {
                "exact": exact_count,
                "inferred": inferred_count,
                "semantic": semantic_count,
                "total_matched": len(matched_skills),
                "total_required": len(jd_required),
            }
        }

    # ── Hiring Recommendation ───────────────────────────────────────────────

    def get_hiring_recommendation(self, score: int, req_missing: int, total_required: int) -> str:
        """Derive a clear hiring recommendation from the match data."""
        if total_required == 0:
            return "Review"
        coverage = (total_required - req_missing) / total_required
        if score >= 80 and coverage >= 0.80:
            return "Strong Hire"
        elif score >= 60 and coverage >= 0.60:
            return "Interview"
        elif score >= 40:
            return "Consider"
        else:
            return "Pass"

    # ── AI Insight Generator ────────────────────────────────────────────────

    def generate_ai_insight(
        self,
        candidate_name: str,
        match_score: int,
        matched_skills: List[Dict],
        missing_skills: List[Dict],
        inferred_skills: List[str],
        years_exp: int
    ) -> str:
        """
        Generate recruiter-quality, multi-sentence assessment text.
        Tone: confident senior technical recruiter, concise, specific.
        """
        name = candidate_name or "The candidate"
        req_missing = [m["skill"] for m in missing_skills if m.get("importance") == "required"]
        nice_missing = [m["skill"] for m in missing_skills if m.get("importance") != "required"]
        top_matches = list({m["jd_skill"] for m in matched_skills})[:5]
        exact_matches = [m["jd_skill"] for m in matched_skills if m.get("match_type") == "exact"][:4]
        semantic_matches = [m["jd_skill"] for m in matched_skills if m.get("match_type") == "semantic"][:3]
        exp_phrase = f"with {years_exp}+ years of experience" if years_exp > 0 else "with an unspecified experience level"

        # Opening assessment
        if match_score >= 80:
            opening = f"{name} stands out as a highly compelling candidate {exp_phrase}, demonstrating strong technical depth across the core requirements."
        elif match_score >= 65:
            opening = f"{name} is a strong potential match {exp_phrase}, covering the majority of the role's technical requirements."
        elif match_score >= 45:
            opening = f"{name} meets several core requirements {exp_phrase} but shows notable gaps in key areas."
        else:
            opening = f"{name} {exp_phrase} currently shows significant skill gaps relative to this role's requirements."

        parts = [opening]

        # Strength sentence
        if exact_matches:
            parts.append(
                f"Verified expertise in {', '.join(exact_matches[:3])} aligns directly with what the role demands."
            )
        elif top_matches:
            parts.append(
                f"Demonstrable coverage across {', '.join(top_matches[:3])} supports role readiness."
            )

        # Semantic / inferred nuance
        if semantic_matches:
            parts.append(
                f"Semantic analysis further surfaces adjacent competency in {', '.join(semantic_matches[:2])}, suggesting transferable depth beyond listed skills."
            )
        if inferred_skills:
            parts.append(
                f"Skill inference reveals implied proficiency in {', '.join(inferred_skills[:3])}, which may not be explicitly stated on the resume."
            )

        # Gap sentence
        if req_missing:
            if len(req_missing) <= 2:
                parts.append(
                    f"The critical gap is {' and '.join(req_missing[:2])} — targeted upskilling in these areas would make this candidate a near-perfect fit."
                )
            else:
                parts.append(
                    f"Key gaps in {', '.join(req_missing[:3])} are required competencies that would need bridging before the candidate is fully role-ready."
                )
        elif match_score >= 65:
            parts.append("All critical technical requirements appear to be covered; this candidate merits priority review.")

        if nice_missing and match_score >= 60:
            parts.append(
                f"Nice-to-have items such as {', '.join(nice_missing[:2])} represent growth opportunities but are not blockers."
            )

        return " ".join(parts)
