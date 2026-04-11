"""
ParserAgent: Extracts structured information from resume files.
Supports PDF, DOCX, and plain text.
"""

import re
import io
from typing import Dict, Any, List


class ParserAgent:
    """
    Agent responsible for parsing resume files into structured data.
    Uses rule-based NLP + regex for speed, LLM-style output format.
    """

    SECTION_HEADERS = {
        "experience": ["experience", "work history", "employment", "career", "professional background"],
        "education": ["education", "academic", "qualification", "degree", "university"],
        "skills": ["skills", "technologies", "tech stack", "competencies", "expertise", "tools"],
        "projects": ["projects", "portfolio", "work samples"],
        "certifications": ["certifications", "certificates", "credentials", "licenses"],
        "summary": ["summary", "objective", "profile", "about", "overview"],
    }

    EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_RE = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
    LINKEDIN_RE = re.compile(r'linkedin\.com/in/[\w\-]+')
    GITHUB_RE = re.compile(r'github\.com/[\w\-]+')

    # Years of experience patterns
    YEARS_RE = re.compile(r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)?', re.IGNORECASE)

    async def parse(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Main parse entry point - routes by file type."""
        
        if filename.endswith('.pdf'):
            text = self._parse_pdf(content)
        elif filename.endswith('.docx'):
            text = self._parse_docx(content)
        else:
            text = content.decode('utf-8', errors='ignore')

        return self._extract_structure(text, filename)

    def _parse_pdf(self, content: bytes) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(content))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except:
                return content.decode('utf-8', errors='ignore')

    def _parse_docx(self, content: bytes) -> str:
        try:
            from docx import Document
            doc = Document(io.BytesIO(content))
            return "\n".join(para.text for para in doc.paragraphs)
        except ImportError:
            return content.decode('utf-8', errors='ignore')

    def _extract_structure(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract structured fields from raw text."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        # --- Contact Info ---
        name = self._extract_name(lines)
        email = self._find_first(self.EMAIL_RE, text)
        phone = self._find_first(self.PHONE_RE, text)
        linkedin = self._find_first(self.LINKEDIN_RE, text)
        github = self._find_first(self.GITHUB_RE, text)

        # --- Sections ---
        sections = self._split_sections(text)

        # --- Skills ---
        raw_skills = self._extract_skills_from_text(
            sections.get("skills", "") + "\n" + sections.get("experience", "") + "\n" + text
        )

        # --- Experience years ---
        years_exp = self._extract_years(text)

        # --- Education ---
        education = self._extract_education(sections.get("education", text))

        # --- Summary ---
        summary = sections.get("summary", lines[0] if lines else "")[:300]

        return {
            "name": name,
            "email": email,
            "phone": phone,
            "linkedin": linkedin,
            "github": github,
            "summary": summary,
            "raw_skills": raw_skills,
            "years_of_experience": years_exp,
            "education": education,
            "sections_detected": list(sections.keys()),
            "raw_text_length": len(text),
            "filename": filename,
            "status": "parsed",
        }

    def _extract_name(self, lines: List[str]) -> str:
        """Heuristic: first non-empty line that looks like a name."""
        for line in lines[:5]:
            # Name: 2-4 words, each capitalized, no special chars except hyphen/apostrophe
            words = line.split()
            if 2 <= len(words) <= 4:
                if all(re.match(r"^[A-Z][a-zA-Z'\-]+$", w) for w in words):
                    return line
        return lines[0] if lines else "Unknown"

    def _find_first(self, pattern, text: str) -> str:
        match = pattern.search(text)
        return match.group(0) if match else ""

    def _split_sections(self, text: str) -> Dict[str, str]:
        """Split resume text into labeled sections."""
        sections = {}
        current_section = "other"
        current_lines = []

        for line in text.split('\n'):
            lower = line.lower().strip()
            matched_section = None
            for section_key, keywords in self.SECTION_HEADERS.items():
                if any(lower == kw or lower.startswith(kw) for kw in keywords):
                    matched_section = section_key
                    break
            if matched_section:
                if current_lines:
                    sections[current_section] = "\n".join(current_lines)
                current_section = matched_section
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections[current_section] = "\n".join(current_lines)

        return sections

    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract potential skill tokens from text using a known skills vocabulary."""
        # Comprehensive skills vocabulary for matching
        KNOWN_SKILLS = [
            # Languages
            "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust", "Ruby", "Swift",
            "Kotlin", "PHP", "Scala", "R", "MATLAB", "Bash", "Shell", "SQL", "HTML", "CSS",
            # Frameworks
            "React", "Vue", "Angular", "Node.js", "Express", "Django", "Flask", "FastAPI",
            "Spring", "Laravel", "Rails", "Next.js", "Nuxt", "Svelte",
            # ML/AI
            "TensorFlow", "PyTorch", "Keras", "scikit-learn", "OpenCV", "NLTK", "spaCy",
            "Hugging Face", "LangChain", "LlamaIndex", "OpenAI", "GPT", "BERT", "Transformers",
            # Data
            "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly", "Spark", "Hadoop", "Kafka",
            "Airflow", "dbt", "Databricks", "Snowflake", "BigQuery", "Redshift",
            # Databases
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
            "SQLite", "DynamoDB", "Neo4j", "InfluxDB",
            # DevOps/Cloud
            "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Terraform", "Ansible",
            "CI/CD", "Jenkins", "GitHub Actions", "GitLab", "Prometheus", "Grafana",
            # APIs/Tools
            "REST", "GraphQL", "gRPC", "FastAPI", "Swagger", "Postman", "Git", "Linux",
            # Soft/Process
            "Agile", "Scrum", "Kanban", "Jira", "Confluence",
        ]

        found = []
        text_lower = text.lower()
        for skill in KNOWN_SKILLS:
            # Match whole word, case-insensitive
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found.append(skill)
        return found

    def _extract_years(self, text: str) -> int:
        """Extract total years of experience."""
        matches = self.YEARS_RE.findall(text)
        if matches:
            return max(int(m) for m in matches)
        # Fallback: count date ranges
        date_ranges = re.findall(r'(20\d\d|19\d\d)\s*[-–]\s*(20\d\d|19\d\d|present|current)', text, re.IGNORECASE)
        if date_ranges:
            total = 0
            import datetime
            current_year = datetime.datetime.now().year
            for start, end in date_ranges:
                s = int(start)
                e = current_year if end.lower() in ('present', 'current') else int(end)
                total += max(0, e - s)
            return min(total, 20)  # cap at 20
        return 0

    def _extract_education(self, text: str) -> List[Dict]:
        """Extract education entries."""
        degrees = ["Bachelor", "Master", "PhD", "B.Tech", "M.Tech", "B.E", "M.E", "B.Sc", "M.Sc", "MBA", "BCA", "MCA"]
        results = []
        for line in text.split('\n'):
            for degree in degrees:
                if degree.lower() in line.lower():
                    results.append({"degree": degree, "line": line.strip()})
                    break
        return results[:3]
