"""
NormalizerAgent: Maps raw extracted skills to a canonical skill taxonomy.
Handles synonyms, abbreviations, and skill hierarchy inference.
"""

from typing import List, Dict, Any, Optional
from agents.knowledge_base import SYNONYMS, TAXONOMY, INFERENCE_RULES, ROLE_MAP


class NormalizerAgent:
    """
    Maintains a hierarchical skill taxonomy and normalizes raw skill strings.
    Also infers implied skills from known relationships.
    """

    def normalize(self, raw_skill: str) -> Dict[str, Any]:
        """Normalize a single raw skill string."""
        key = raw_skill.lower().strip()
        canonical = SYNONYMS.get(key, raw_skill)

        # Find category
        category, subcategory = "Other", "Miscellaneous"
        for cat, subcats in TAXONOMY.items():
            for subcat, skills in subcats.items():
                # Case insensitive check for known taxonomy skills
                if any(s.lower() == canonical.lower() for s in skills):
                    category = cat
                    subcategory = subcat
                    break
            if category != "Other":
                break

        return {
            "raw": raw_skill,
            "canonical": canonical,
            "category": category,
            "subcategory": subcategory,
            "confidence": 1.0 if key in SYNONYMS else 0.85,
        }

    def normalize_batch(self, skills: List[str]) -> List[Dict[str, Any]]:
        return [self.normalize(s) for s in skills]

    def infer_skills(self, known_skills: List[str]) -> List[str]:
        """Infer additional skills from known skills using inference rules."""
        inferred = set()
        canonical_skills = {self.normalize(s)["canonical"] for s in known_skills}
        
        # We need case mapping because inference rules use exact capitalization
        # Let's do a case-insensitive lookup
        skill_lower_map = {k.lower(): v for k, v in INFERENCE_RULES.items()}

        for skill in canonical_skills:
            implied = skill_lower_map.get(skill.lower(), [])
            for impl in implied:
                # Add only if not already in canonical skills (case insensitive check)
                if not any(impl.lower() == c.lower() for c in canonical_skills):
                    inferred.add(impl)

        return list(inferred)

    def suggest_roles(self, canonical_skills: List[str]) -> List[str]:
        """Suggest job roles based on skill profile."""
        skill_set = {s.lower() for s in canonical_skills}
        matching_roles = []

        for role_def in ROLE_MAP:
            signals = [s.lower() for s in role_def["required_signals"]]
            matched = sum(1 for s in signals if s in skill_set)
            if matched >= role_def["min_match"]:
                score = matched / len(signals)
                matching_roles.append((role_def["role"], score))

        # Sort by match quality
        matching_roles.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in matching_roles[:3]]

    def get_taxonomy(self) -> Dict:
        return TAXONOMY

    def get_learning_recommendations(self, missing_skills: List[str]) -> List[str]:
        """Generate learning path recommendations for missing skills."""
        recs = []
        resource_map = {
            "deep learning":        "Fast.ai Practical Deep Learning (free) → fast.ai",
            "large language models":"Hugging Face NLP Course (free) → huggingface.co/learn",
            "kubernetes":           "KCNA certification path → kubernetes.io/docs",
            "aws":                  "AWS Cloud Practitioner (beginner) → aws.amazon.com/training",
            "machine learning":     "Andrew Ng's ML Specialization → coursera.org",
            "docker":               "Docker Getting Started official tutorial → docs.docker.com",
            "react":                "React official docs + Scrimba React course → scrimba.com",
            "apache spark":         "Databricks free Spark training → training.databricks.com",
            "dbt":                  "dbt Learn platform (free) → courses.getdbt.com",
            "graphql":              "The Odin Project / Apollo Docs → apollographql.com/docs",
            "python":               "100 Days of Code: The Complete Python Pro Bootcamp → udemy.com",
            "sql":                  "SQL for Data Science → coursera.org",
            "javascript":           "JavaScript: Understanding the Weird Parts → udemy.com"
        }
        for skill in missing_skills[:5]:
            desc = resource_map.get(skill.lower())
            if desc:
                recs.append(f"Learn {skill}: {desc}")
            else:
                recs.append(f"Explore {skill}: Search '{skill} tutorial' on YouTube or Coursera")
        return recs
