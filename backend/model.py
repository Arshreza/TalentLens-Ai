"""
model.py — Semantic similarity engine using SentenceTransformers.
Uses section-aware embedding: skills, experience, and education are
weighted separately for a more accurate match score.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load once at startup (cached in memory for all subsequent requests)
model = SentenceTransformer('all-MiniLM-L6-v2')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

SECTION_HEADERS = {
    "skills":      r"(skills|technologies|tech stack|tools|competencies|expertise)",
    "experience":  r"(experience|work history|employment|projects|professional background)",
    "education":   r"(education|qualification|degree|academic|university|college)",
    "summary":     r"(summary|objective|profile|about me|overview)",
}

def extract_section(text: str, section_key: str) -> str:
    """
    Extract a specific section from resume/JD text by header detection.
    Falls back to full text if section not found.
    """
    pattern = SECTION_HEADERS.get(section_key, "")
    if not pattern:
        return text

    lines = text.split('\n')
    start_idx = -1
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if re.search(pattern, line, re.IGNORECASE):
            start_idx = i
            break

    if start_idx == -1:
        return ""

    # Stop at next section header
    all_headers = '|'.join(SECTION_HEADERS.values())
    for i in range(start_idx + 1, len(lines)):
        if re.search(all_headers, lines[i], re.IGNORECASE) and i > start_idx + 1:
            end_idx = i
            break

    return '\n'.join(lines[start_idx:end_idx]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING & SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def embed(text: str):
    """Encode text into a semantic vector."""
    if not text or not text.strip():
        return None
    return model.encode(text.strip())


def cosine_sim(a, b) -> float:
    """Compute cosine similarity between two vectors. Returns 0 if either is None."""
    if a is None or b is None:
        return 0.0
    return float(cosine_similarity([a], [b])[0][0])


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTED MATCH SCORE
# ─────────────────────────────────────────────────────────────────────────────

SECTION_WEIGHTS = {
    "skills":     0.45,   # Skills are the most important signal
    "experience": 0.35,   # Work experience is second
    "summary":    0.10,   # Summary/objective has minor weight
    "full":       0.10,   # Full-text semantic backup
}

def get_match_score(resume_text: str, jd_text: str) -> tuple[int, float]:
    """
    Compute a weighted, section-aware match score between resume and JD.

    Returns:
        (score_0_to_100: int, confidence_0_to_1: float)
    """
    total_score = 0.0
    total_weight = 0.0

    # Compare each section
    for section, weight in SECTION_WEIGHTS.items():
        if section == "full":
            r_vec = embed(resume_text)
            jd_vec = embed(jd_text)
        else:
            r_section = extract_section(resume_text, section) or resume_text
            jd_section = extract_section(jd_text, section) or jd_text
            r_vec = embed(r_section)
            jd_vec = embed(jd_section)

        sim = cosine_sim(r_vec, jd_vec)
        total_score += sim * weight
        total_weight += weight

    # Normalise
    if total_weight == 0:
        return 0, 0.0

    raw_score = total_score / total_weight

    # Rescale cosine similarity to a human-friendly 0-100 score.
    # Typical SentenceTransformer cosine values for resume/JD pairs:
    #   Strong match  → 0.72-0.88  →  should read as 75-100
    #   Good match    → 0.60-0.72  →  should read as 55-75
    #   Moderate      → 0.50-0.60  →  should read as 30-55
    #   Weak / unrelated → <0.50   →  should read as 0-30
    # Formula: (raw - 0.38) / 0.47  → clips at 0 below baseline, caps at 100
    rescaled = max(0.0, (raw_score - 0.38) / 0.47)
    final_score = int(round(min(rescaled, 1.0) * 100))

    # Confidence = certainty of the verdict (very high or very low = confident)
    confidence = 0.50 + abs(rescaled - 0.50) * 0.6
    confidence = round(min(confidence, 1.0), 4)

    return final_score, confidence