"""
Knowledge Base: The comprehensive data source for taxonomies, skills, synonyms,
inference rules, and roles used across all AI agents.
"""

# ── 1. Synonyms & Misspellings Map ──────────────────────────────────────────
SYNONYMS = {
    # JavaScript & Web
    "js": "JavaScript", "javascript": "JavaScript", "es6": "JavaScript", "ecmascript": "JavaScript",
    "ts": "TypeScript", "typescript": "TypeScript",
    "react": "React", "reactjs": "React", "react.js": "React",
    "vue": "Vue.js", "vuejs": "Vue.js", "vue.js": "Vue.js",
    "angular": "Angular", "angularjs": "Angular", "angular.js": "Angular",
    "node": "Node.js", "nodejs": "Node.js", "node.js": "Node.js",
    "next": "Next.js", "nextjs": "Next.js", "next.js": "Next.js",
    "nuxt": "Nuxt.js", "nuxtjs": "Nuxt.js",
    "jquery": "jQuery", "tailwind": "Tailwind CSS", "tailwindcss": "Tailwind CSS",
    "html5": "HTML", "css3": "CSS",

    # Backend & Languages
    "py": "Python", "python3": "Python",
    "golang": "Go", "go lang": "Go",
    "cpp": "C++", "c++11": "C++", "c++14": "C++", "c++17": "C++",
    "c#": "C#", "csharp": "C#", "dotnet": ".NET", ".net core": ".NET",
    "jv": "Java", "j2ee": "Java",
    "rb": "Ruby", "ror": "Ruby on Rails", "rails": "Ruby on Rails",
    "php8": "PHP", "php7": "PHP",
    
    # ML & AI
    "sklearn": "scikit-learn", "scikit": "scikit-learn",
    "tf": "TensorFlow", "tensorflow": "TensorFlow",
    "torch": "PyTorch", "pytorch": "PyTorch",
    "hf": "Hugging Face", "huggingface": "Hugging Face",
    "llm": "Large Language Models", "large language models": "Large Language Models",
    "llms": "Large Language Models",
    "nlp": "Natural Language Processing", "natural language processing": "Natural Language Processing",
    "cv": "Computer Vision", "computer vision": "Computer Vision",
    "ml": "Machine Learning", "machine learning": "Machine Learning",
    "dl": "Deep Learning", "deep learning": "Deep Learning",
    "genai": "Generative AI", "generative ai": "Generative AI",
    "rag": "Retrieval-Augmented Generation",

    # Cloud & DevOps
    "gcp": "Google Cloud", "google cloud platform": "Google Cloud",
    "aws": "AWS", "amazon web services": "AWS",
    "azure": "Microsoft Azure", "microsoft azure": "Microsoft Azure",
    "k8s": "Kubernetes", "kube": "Kubernetes",
    "gh actions": "GitHub Actions", "github-actions": "GitHub Actions",
    "ci/cd": "CI/CD", "cicd": "CI/CD", "ci cd": "CI/CD",
    "tfrm": "Terraform", "docker-compose": "Docker",
    
    # DB & Data
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "pgsql": "PostgreSQL",
    "mongo": "MongoDB", "mongodb": "MongoDB",
    "elastic": "Elasticsearch", "elk": "Elasticsearch",
    "dynamo": "DynamoDB", "dynamodb": "DynamoDB",
    "sql server": "Microsoft SQL Server", "mssql": "Microsoft SQL Server",
    "bq": "BigQuery", "big query": "BigQuery",
    
    # Mobile
    "rn": "React Native", "react-native": "React Native",
    
    # Architecture
    "sql": "SQL", "nosql": "NoSQL",
    "rest": "REST APIs", "restful": "REST APIs", "rest api": "REST APIs",
    "graphql": "GraphQL", "grpc": "gRPC",
    "git": "Git", "github": "GitHub", "gitlab": "GitLab",
    "linux": "Linux", "unix": "Linux", "ubuntu": "Linux",
    "agile": "Agile", "scrum": "Scrum",
    "microservices": "Microservices"
}

# ── 2. Master Taxonomy (Category → Subcategory → Skills) ───────────────────
TAXONOMY = {
    "Technical Skills": {
        "Programming Languages": [
            "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go",
            "Rust", "Ruby", "Swift", "Kotlin", "PHP", "Scala", "R", "SQL",
            "Dart", "Objective-C", "Bash", "PowerShell", "Lua"
        ],
        "Frontend Development": [
            "React", "Vue.js", "Angular", "Next.js", "Nuxt.js", "Svelte", 
            "HTML", "CSS", "jQuery", "Tailwind CSS", "Bootstrap", "Material UI", 
            "WebAssembly", "Redux", "Webpack", "Vite"
        ],
        "Backend & Frameworks": [
            "Node.js", "Express", "Django", "Flask", "FastAPI", "Spring", "Spring Boot", 
            "Laravel", "Ruby on Rails", ".NET", "ASP.NET", "NestJS", "GraphQL", "REST APIs",
            "gRPC", "Microservices", "Event-Driven Architecture"
        ],
        "Mobile Development": [
            "iOS", "Android", "React Native", "Flutter", "SwiftUI", "Jetpack Compose", "Xamarin"
        ],
        "Databases": [
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra", 
            "SQLite", "DynamoDB", "Neo4j", "InfluxDB", "Snowflake", "Microsoft SQL Server",
            "Oracle", "MariaDB", "Cosmos DB", "Bigtable"
        ],
        "Cloud & DevOps": [
            "AWS", "Google Cloud", "Microsoft Azure", "Docker", "Kubernetes",
            "Terraform", "Ansible", "CI/CD", "GitHub Actions", "Jenkins", "GitLab CI",
            "Prometheus", "Grafana", "Datadog", "Pulumi", "Serverless Framework"
        ],
        "Cybersecurity": [
            "Penetration Testing", "Cryptography", "OWASP", "IAM", "Cloud Security",
            "Network Security", "Vulnerability Management", "SIEM", "Firewalls"
        ]
    },
    "AI & Data Science": {
        "Machine Learning": [
            "Machine Learning", "scikit-learn", "XGBoost", "LightGBM",
            "Feature Engineering", "Model Evaluation"
        ],
        "Deep Learning": [
            "Deep Learning", "TensorFlow", "PyTorch", "Keras", "Neural Networks",
            "CNNs", "RNNs", "Transformers"
        ],
        "NLP & GenAI": [
            "Natural Language Processing", "Large Language Models", "Hugging Face",
            "LangChain", "Retrieval-Augmented Generation", "OpenAI", "Generative AI",
            "spaCy", "NLTK", "Embeddings"
        ],
        "Computer Vision": [
            "Computer Vision", "OpenCV", "Image Processing", "Object Detection", "YOLO"
        ],
        "Data Engineering": [
            "Apache Spark", "Kafka", "Airflow", "dbt", "Databricks", "BigQuery",
            "Pandas", "NumPy", "ETL", "Hadoop", "Flink", "Redshift"
        ],
    },
    "Soft Skills & Management": {
        "Methodologies": ["Agile", "Scrum", "Kanban", "Jira", "Confluence"],
        "Leadership": ["Team Leadership", "Mentoring", "Project Management", "Product Management"],
        "Communication": ["Technical Writing", "Presentation", "Stakeholder Management", "Cross-functional Collaboration"],
        "Design": ["UI/UX Design", "Figma", "User Research", "Wireframing"]
    }
}

# ── 3. Flattened Skills List for Parsers ───────────────────────────────────
# We generate this automatically from TAXONOMY and SYNONYMS.
KNOWN_SKILLS = set(list(SYNONYMS.keys()))
for main_cat, subcats in TAXONOMY.items():
    for subcat, skills in subcats.items():
        for skill in skills:
            KNOWN_SKILLS.add(skill)
KNOWN_SKILLS = sorted(list(KNOWN_SKILLS))

# ── 4. Deep Inference Rules Graph (If X -> Add Y) ──────────────────────────
INFERENCE_RULES = {
    # Framework to Language
    "React":            ["JavaScript", "Frontend Development", "HTML", "CSS"],
    "Next.js":          ["React", "JavaScript", "TypeScript", "Frontend Development", "SSR"],
    "Angular":          ["TypeScript", "Frontend Development"],
    "Vue.js":           ["JavaScript", "Frontend Development"],
    "Node.js":          ["JavaScript", "Backend Development"],
    "Express":          ["Node.js", "JavaScript", "Backend Development"],
    "NestJS":           ["Node.js", "TypeScript", "Backend Development"],
    "Django":           ["Python", "REST APIs", "Backend Development", "SQL"],
    "Flask":            ["Python", "Backend Development"],
    "FastAPI":          ["Python", "REST APIs", "Backend Development"],
    "Spring Boot":      ["Java", "Spring", "Backend Development", "Microservices"],
    "Ruby on Rails":    ["Ruby", "Backend Development"],
    "Laravel":          ["PHP", "Backend Development"],
    
    # ML & AI
    "TensorFlow":       ["Deep Learning", "Machine Learning", "Neural Networks", "Python"],
    "PyTorch":          ["Deep Learning", "Machine Learning", "Neural Networks", "Python"],
    "Keras":            ["Deep Learning", "TensorFlow", "Machine Learning", "Python"],
    "scikit-learn":     ["Machine Learning", "Python", "Data Science"],
    "LangChain":        ["Large Language Models", "Natural Language Processing", "Python", "Retrieval-Augmented Generation", "Generative AI"],
    "Hugging Face":     ["Transformers", "Natural Language Processing", "Deep Learning", "Large Language Models", "Python"],
    "OpenAI":           ["Large Language Models", "Generative AI", "APIs"],
    "Pandas":           ["Python", "Data Analysis", "Data Science"],
    "NumPy":            ["Python", "Data Science"],
    
    # Cloud & DevOps
    "Kubernetes":       ["Docker", "Cloud computing", "DevOps", "Microservices", "Containers"],
    "Docker":           ["Linux", "DevOps", "Containers"],
    "Terraform":        ["Infrastructure as Code", "DevOps", "Cloud Computing"],
    "AWS":              ["Cloud Computing", "DevOps", "Distributed Systems"],
    "Google Cloud":     ["Cloud Computing", "DevOps"],
    "Microsoft Azure":  ["Cloud Computing", "DevOps"],
    "GitHub Actions":   ["CI/CD", "DevOps", "Git"],
    "Jenkins":          ["CI/CD", "DevOps"],
    
    # Data Engineering
    "Apache Spark":     ["Big Data", "Distributed Computing", "Data Engineering"],
    "Kafka":            ["Event-Driven Architecture", "Distributed Systems", "Data Engineering", "Streaming"],
    "Airflow":          ["Data Engineering", "Python", "ETL", "Orchestration"],
    "dbt":              ["Data Engineering", "SQL", "Data Warehousing", "ETL"],
    "Snowflake":        ["Data Warehousing", "SQL", "Cloud Computing"],
    "BigQuery":         ["Data Warehousing", "SQL", "Google Cloud"],
    
    # Mobile
    "React Native":     ["JavaScript", "Mobile Development", "React"],
    "Flutter":          ["Dart", "Mobile Development"],
    "SwiftUI":          ["Swift", "iOS", "Mobile Development"],
    "Jetpack Compose":  ["Kotlin", "Android", "Mobile Development"],
    
    # Database
    "PostgreSQL":       ["SQL", "Relational Databases"],
    "MySQL":            ["SQL", "Relational Databases"],
    "MongoDB":          ["NoSQL", "Databases"],
    "Redis":            ["Caching", "NoSQL", "In-Memory Databases"],
    "Elasticsearch":    ["Search Engineering", "NoSQL", "Distributed Systems"],
    
    # General
    "GraphQL":          ["APIs", "Backend Development"],
    "REST APIs":        ["APIs", "Backend Development", "HTTP"],
    "Microservices":    ["Distributed Systems", "Backend Architecture"]
}

# ── 5. Rich Role Mapping ───────────────────────────────────────────────────
ROLE_MAP = [
    {
        "role": "Senior ML Engineer",
        "required_signals": ["PyTorch", "TensorFlow", "Machine Learning", "Python", "Deep Learning"],
        "min_match": 3
    },
    {
        "role": "GenAI / LLM Engineer",
        "required_signals": ["LangChain", "Large Language Models", "Hugging Face", "Retrieval-Augmented Generation", "OpenAI", "Generative AI"],
        "min_match": 2
    },
    {
        "role": "Data Scientist",
        "required_signals": ["Python", "scikit-learn", "Pandas", "NumPy", "Machine Learning", "Data Analysis", "SQL"],
        "min_match": 3
    },
    {
        "role": "Data Engineer",
        "required_signals": ["Apache Spark", "Kafka", "Airflow", "dbt", "SQL", "BigQuery", "ETL", "Python", "Data Warehousing"],
        "min_match": 3
    },
    {
        "role": "Backend Engineer",
        "required_signals": ["Python", "FastAPI", "Django", "PostgreSQL", "Docker", "REST APIs", "Node.js", "Java", "Go", "Microservices"],
        "min_match": 3
    },
    {
        "role": "Frontend Developer",
        "required_signals": ["React", "Vue.js", "TypeScript", "Next.js", "CSS", "HTML", "JavaScript", "Tailwind CSS"],
        "min_match": 3
    },
    {
        "role": "Full Stack Developer",
        "required_signals": ["React", "Node.js", "PostgreSQL", "REST APIs", "JavaScript", "TypeScript", "CSS", "Frontend Development", "Backend Development"],
        "min_match": 4
    },
    {
        "role": "DevOps Engineer",
        "required_signals": ["Kubernetes", "Docker", "Terraform", "CI/CD", "AWS", "Linux", "Jenkins", "GitHub Actions", "Infrastructure as Code"],
        "min_match": 3
    },
    {
        "role": "Cloud Architect",
        "required_signals": ["AWS", "Google Cloud", "Microsoft Azure", "Terraform", "Kubernetes", "Microservices", "Distributed Systems", "System Design"],
        "min_match": 3
    },
    {
        "role": "Mobile Developer",
        "required_signals": ["iOS", "Android", "React Native", "Flutter", "Swift", "Kotlin", "Mobile Development"],
        "min_match": 2
    },
    {
        "role": "Cybersecurity Engineer",
        "required_signals": ["Penetration Testing", "Network Security", "Cryptography", "OWASP", "IAM", "Cloud Security"],
        "min_match": 2
    },
    {
        "role": "Product Manager",
        "required_signals": ["Agile", "Scrum", "Product Management", "Jira", "Cross-functional Collaboration", "Stakeholder Management"],
        "min_match": 3
    },
    {
        "role": "UI/UX Designer",
        "required_signals": ["UI/UX Design", "Figma", "User Research", "Wireframing"],
        "min_match": 2
    }
]
