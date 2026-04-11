"""
NormalizerAgent: Maps raw extracted skills to a canonical skill taxonomy.
Handles synonyms, abbreviations, and skill hierarchy inference.
"""

from typing import List, Dict, Any, Optional


class NormalizerAgent:
    """
    Maintains a hierarchical skill taxonomy and normalizes raw skill strings.
    Also infers implied skills from known relationships.
    """

    # ── Synonym / Alias Map ──────────────────────────────────────────────────
    SYNONYMS: Dict[str, str] = {
        # JavaScript ecosystem
        "js": "JavaScript", "javascript": "JavaScript", "es6": "JavaScript", "es2015": "JavaScript",
        "ts": "TypeScript", "typescript": "TypeScript",
        "react": "React", "reactjs": "React", "react.js": "React",
        "vue": "Vue.js", "vuejs": "Vue.js", "vue.js": "Vue.js",
        "angular": "Angular", "angularjs": "Angular",
        "node": "Node.js", "nodejs": "Node.js", "node.js": "Node.js",
        "next": "Next.js", "nextjs": "Next.js",
        # Python
        "py": "Python", "python3": "Python", "python2": "Python",
        "sklearn": "scikit-learn", "scikit": "scikit-learn",
        "tf": "TensorFlow", "tensorflow": "TensorFlow",
        "torch": "PyTorch", "pytorch": "PyTorch",
        "hf": "Hugging Face", "huggingface": "Hugging Face",
        # Containers / DevOps
        "k8s": "Kubernetes", "kube": "Kubernetes",
        "gh actions": "GitHub Actions", "github-actions": "GitHub Actions",
        "ci/cd": "CI/CD", "cicd": "CI/CD",
        # Databases
        "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
        "mongo": "MongoDB", "mongodb": "MongoDB",
        "elastic": "Elasticsearch", "elk": "Elasticsearch",
        "dynamo": "DynamoDB", "dynamodb": "DynamoDB",
        # Cloud
        "gcp": "Google Cloud", "google cloud platform": "Google Cloud",
        "aws": "AWS", "amazon web services": "AWS",
        "azure": "Microsoft Azure", "microsoft azure": "Microsoft Azure",
        # ML
        "nlp": "Natural Language Processing", "natural language processing": "Natural Language Processing",
        "cv": "Computer Vision", "computer vision": "Computer Vision",
        "dl": "Deep Learning", "deep learning": "Deep Learning",
        "ml": "Machine Learning", "machine learning": "Machine Learning",
        "llm": "Large Language Models", "large language models": "Large Language Models",
        "rag": "Retrieval-Augmented Generation",
        # Other
        "sql": "SQL", "nosql": "NoSQL",
        "rest": "REST APIs", "restful": "REST APIs", "graphql": "GraphQL",
        "git": "Git", "github": "GitHub", "gitlab": "GitLab",
        "linux": "Linux", "unix": "Linux",
        "agile": "Agile", "scrum": "Scrum",
    }

    # ── Skill Hierarchy (Category → Skills) ────────────────────────────────
    TAXONOMY: Dict[str, Dict[str, List[str]]] = {
        "Technical Skills": {
            "Programming Languages": [
                "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go",
                "Rust", "Ruby", "Swift", "Kotlin", "PHP", "Scala", "R", "SQL"
            ],
            "Web Frameworks": [
                "React", "Vue.js", "Angular", "Next.js", "Node.js", "Express",
                "Django", "Flask", "FastAPI", "Spring", "Laravel", "Rails"
            ],
            "Databases": [
                "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
                "Cassandra", "SQLite", "DynamoDB", "Neo4j", "InfluxDB", "Snowflake"
            ],
            "Cloud & DevOps": [
                "AWS", "Google Cloud", "Microsoft Azure", "Docker", "Kubernetes",
                "Terraform", "Ansible", "CI/CD", "GitHub Actions", "Jenkins"
            ],
            "APIs & Architecture": [
                "REST APIs", "GraphQL", "gRPC", "Microservices", "Event-Driven Architecture"
            ],
        },
        "AI & Data Science": {
            "Machine Learning": [
                "Machine Learning", "scikit-learn", "XGBoost", "LightGBM",
                "Feature Engineering", "Model Evaluation"
            ],
            "Deep Learning": [
                "Deep Learning", "TensorFlow", "PyTorch", "Keras", "Neural Networks",
                "CNNs", "RNNs", "Transformers", "BERT", "GPT"
            ],
            "NLP & LLMs": [
                "Natural Language Processing", "Large Language Models", "Hugging Face",
                "LangChain", "Retrieval-Augmented Generation", "Embeddings", "spaCy", "NLTK"
            ],
            "Computer Vision": [
                "Computer Vision", "OpenCV", "Image Processing", "Object Detection", "YOLO"
            ],
            "Data Engineering": [
                "Apache Spark", "Kafka", "Airflow", "dbt", "Databricks", "BigQuery",
                "Pandas", "NumPy", "ETL"
            ],
        },
        "Soft Skills": {
            "Leadership": ["Team Leadership", "Mentoring", "Project Management"],
            "Collaboration": ["Agile", "Scrum", "Kanban", "Cross-functional Collaboration"],
            "Communication": ["Technical Writing", "Presentation", "Stakeholder Management"],
        }
    }

    # ── Inference Rules (if you have X → you likely have Y) ────────────────
    INFERENCE_RULES: Dict[str, List[str]] = {
        "TensorFlow":       ["Deep Learning", "Machine Learning", "Neural Networks", "Python"],
        "PyTorch":          ["Deep Learning", "Machine Learning", "Neural Networks", "Python"],
        "Keras":            ["Deep Learning", "TensorFlow", "Machine Learning"],
        "scikit-learn":     ["Machine Learning", "Python"],
        "LangChain":        ["Large Language Models", "Natural Language Processing", "Python", "Retrieval-Augmented Generation"],
        "Hugging Face":     ["Transformers", "Natural Language Processing", "Deep Learning", "Large Language Models"],
        "Kubernetes":       ["Docker", "DevOps", "Microservices"],
        "Docker":           ["Linux", "DevOps"],
        "Apache Spark":     ["Big Data", "Python", "Distributed Computing"],
        "React":            ["JavaScript", "Frontend Development", "HTML", "CSS"],
        "Next.js":          ["React", "JavaScript", "TypeScript", "SSR"],
        "FastAPI":          ["Python", "REST APIs", "Backend Development"],
        "Django":           ["Python", "REST APIs", "Backend Development", "SQL"],
        "AWS":              ["Cloud Computing", "DevOps"],
        "Google Cloud":     ["Cloud Computing", "DevOps"],
        "Elasticsearch":    ["Search Engineering", "NoSQL"],
        "Natural Language Processing": ["Machine Learning", "Python"],
        "Computer Vision":  ["Deep Learning", "Python", "Image Processing"],
        "GraphQL":          ["APIs", "Backend Development"],
    }

    # ── Role Suggestions based on skill clusters ───────────────────────────
    ROLE_MAP = [
        {
            "role": "Senior ML Engineer",
            "required_signals": ["PyTorch", "TensorFlow", "Machine Learning", "Python"],
            "min_match": 2
        },
        {
            "role": "LLM / GenAI Engineer",
            "required_signals": ["LangChain", "Large Language Models", "Hugging Face", "Retrieval-Augmented Generation", "OpenAI"],
            "min_match": 2
        },
        {
            "role": "Full Stack Developer",
            "required_signals": ["React", "Node.js", "PostgreSQL", "REST APIs", "JavaScript"],
            "min_match": 3
        },
        {
            "role": "Backend Engineer",
            "required_signals": ["Python", "FastAPI", "Django", "PostgreSQL", "Docker", "REST APIs"],
            "min_match": 3
        },
        {
            "role": "Data Engineer",
            "required_signals": ["Apache Spark", "Kafka", "Airflow", "dbt", "SQL", "BigQuery"],
            "min_match": 2
        },
        {
            "role": "DevOps / Platform Engineer",
            "required_signals": ["Kubernetes", "Docker", "Terraform", "CI/CD", "AWS"],
            "min_match": 2
        },
        {
            "role": "Data Scientist",
            "required_signals": ["Python", "scikit-learn", "Pandas", "NumPy", "Machine Learning"],
            "min_match": 3
        },
        {
            "role": "Frontend Developer",
            "required_signals": ["React", "Vue.js", "TypeScript", "Next.js", "CSS"],
            "min_match": 2
        },
    ]

    def normalize(self, raw_skill: str) -> Dict[str, Any]:
        """Normalize a single raw skill string."""
        key = raw_skill.lower().strip()
        canonical = self.SYNONYMS.get(key, raw_skill)

        # Find category
        category, subcategory = "Other", "Miscellaneous"
        for cat, subcats in self.TAXONOMY.items():
            for subcat, skills in subcats.items():
                if canonical in skills:
                    category = cat
                    subcategory = subcat
                    break

        return {
            "raw": raw_skill,
            "canonical": canonical,
            "category": category,
            "subcategory": subcategory,
            "confidence": 1.0 if key in self.SYNONYMS else 0.85,
        }

    def normalize_batch(self, skills: List[str]) -> List[Dict[str, Any]]:
        return [self.normalize(s) for s in skills]

    def infer_skills(self, known_skills: List[str]) -> List[str]:
        """Infer additional skills from known skills using inference rules."""
        inferred = set()
        canonical_skills = {self.normalize(s)["canonical"] for s in known_skills}

        for skill in canonical_skills:
            implied = self.INFERENCE_RULES.get(skill, [])
            for impl in implied:
                if impl not in canonical_skills:
                    inferred.add(impl)

        return list(inferred)

    def suggest_roles(self, canonical_skills: List[str]) -> List[str]:
        """Suggest job roles based on skill profile."""
        skill_set = set(canonical_skills)
        matching_roles = []

        for role_def in self.ROLE_MAP:
            signals = role_def["required_signals"]
            matched = sum(1 for s in signals if s in skill_set)
            if matched >= role_def["min_match"]:
                score = matched / len(signals)
                matching_roles.append((role_def["role"], score))

        # Sort by match quality
        matching_roles.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in matching_roles[:3]]

    def get_taxonomy(self) -> Dict:
        return self.TAXONOMY

    def get_learning_recommendations(self, missing_skills: List[str]) -> List[str]:
        """Generate learning path recommendations for missing skills."""
        recs = []
        resource_map = {
            "Deep Learning":        "Fast.ai Practical Deep Learning (free) → fast.ai",
            "Large Language Models":"Hugging Face NLP Course (free) → huggingface.co/learn",
            "Kubernetes":           "KCNA certification path → kubernetes.io/docs",
            "AWS":                  "AWS Cloud Practitioner (beginner) → aws.amazon.com/training",
            "Machine Learning":     "Andrew Ng's ML Specialization → coursera.org",
            "Docker":               "Docker Getting Started official tutorial → docs.docker.com",
            "React":                "React official docs + Scrimba React course → scrimba.com",
            "Apache Spark":         "Databricks free Spark training → training.databricks.com",
            "dbt":                  "dbt Learn platform (free) → courses.getdbt.com",
            "GraphQL":              "The Odin Project / Apollo Docs → apollographql.com/docs",
        }
        for skill in missing_skills[:5]:
            if skill in resource_map:
                recs.append(f"Learn {skill}: {resource_map[skill]}")
            else:
                recs.append(f"Explore {skill}: Search '{skill} tutorial' on YouTube or Coursera")
        return recs
