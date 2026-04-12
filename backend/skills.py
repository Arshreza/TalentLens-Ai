"""
skills.py — Comprehensive skill taxonomy with aliases, categories, and inference rules.
Replaces the original 13-skill hardcoded list with 200+ skills across all domains.
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# MASTER SKILLS TAXONOMY
# Key   = canonical skill name (shown in UI)
# Value = list of aliases/variants to match in text (all lowercase)
# ─────────────────────────────────────────────────────────────────────────────
SKILLS_TAXONOMY = {
    # ── Programming Languages ───────────────────────────────────────────────
    "Python":           ["python", "python3", "py"],
    "JavaScript":       ["javascript", "js", "es6", "es2015", "ecmascript"],
    "TypeScript":       ["typescript", "ts"],
    "Java":             ["java", "java8", "java11", "java17"],
    "C++":              ["c++", "cpp", "c plus plus"],
    "C#":               ["c#", "csharp", "c sharp", ".net"],
    "Go":               ["golang", "go lang", " go "],
    "Rust":             ["rust", "rust-lang"],
    "Kotlin":           ["kotlin"],
    "Swift":            ["swift", "swiftui"],
    "Ruby":             ["ruby", "ruby on rails", "rails"],
    "PHP":              ["php", "laravel", "symfony"],
    "Scala":            ["scala", "akka"],
    "R":                [" r ", "r programming", "rstudio", "tidyverse"],
    "MATLAB":           ["matlab"],
    "Bash/Shell":       ["bash", "shell scripting", "shell script", "zsh", "powershell"],
    "SQL":              ["sql", "mysql", "t-sql", "pl/sql", "plsql"],

    # ── Web Frontend ────────────────────────────────────────────────────────
    "React":            ["react", "reactjs", "react.js", "react native"],
    "Vue.js":           ["vue", "vuejs", "vue.js", "vuex"],
    "Angular":          ["angular", "angularjs", "angular.js"],
    "Next.js":          ["next.js", "nextjs"],
    "HTML/CSS":         ["html", "css", "html5", "css3", "sass", "scss", "less"],
    "Tailwind CSS":     ["tailwind", "tailwindcss"],
    "Redux":            ["redux", "redux-toolkit", "zustand"],
    "GraphQL":          ["graphql", "apollo", "apollo client"],
    "WebSocket":        ["websocket", "socket.io", "ws"],

    # ── Web Backend ─────────────────────────────────────────────────────────
    "FastAPI":          ["fastapi", "fast api"],
    "Django":           ["django", "django rest framework", "drf"],
    "Flask":            ["flask"],
    "Node.js":          ["node.js", "nodejs", "express", "expressjs"],
    "Spring Boot":      ["spring boot", "spring", "springboot"],
    "REST API":         ["rest api", "restful", "rest", "api development", "http api"],
    "gRPC":             ["grpc", "protobuf", "protocol buffers"],

    # ── Databases ───────────────────────────────────────────────────────────
    "PostgreSQL":       ["postgresql", "postgres", "psql"],
    "MySQL":            ["mysql"],
    "MongoDB":          ["mongodb", "mongo", "mongoose"],
    "Redis":            ["redis", "memcached"],
    "Elasticsearch":    ["elasticsearch", "elastic search", "opensearch"],
    "SQLite":           ["sqlite"],
    "Cassandra":        ["cassandra", "apache cassandra"],
    "DynamoDB":         ["dynamodb", "dynamo db"],
    "Snowflake":        ["snowflake"],
    "BigQuery":         ["bigquery", "big query"],

    # ── Cloud & DevOps ──────────────────────────────────────────────────────
    "AWS":              ["aws", "amazon web services", "ec2", "s3", "lambda", "sagemaker", "eks"],
    "Google Cloud":     ["gcp", "google cloud", "google cloud platform", "vertex ai", "bigquery", "gke"],
    "Azure":            ["azure", "microsoft azure", "azure ml", "aks"],
    "Docker":           ["docker", "dockerfile", "docker-compose", "containerization"],
    "Kubernetes":       ["kubernetes", "k8s", "kubectl", "helm", "eks", "aks", "gke"],
    "Terraform":        ["terraform", "iac", "infrastructure as code"],
    "CI/CD":            ["ci/cd", "cicd", "github actions", "gitlab ci", "jenkins", "circleci", "travis ci", "continuous integration", "continuous delivery"],
    "Ansible":          ["ansible"],
    "Linux":            ["linux", "ubuntu", "centos", "debian", "unix"],
    "Nginx":            ["nginx", "apache", "load balancer"],
    "Kafka":            ["kafka", "apache kafka", "event streaming", "message queue", "rabbitmq", "pubsub"],

    # ── Machine Learning & AI ───────────────────────────────────────────────
    "Machine Learning": ["machine learning", "ml", "supervised learning", "unsupervised learning"],
    "Deep Learning":    ["deep learning", "dl", "neural network", "neural networks", "ann", "cnn", "rnn", "lstm", "transformer"],
    "NLP":              ["nlp", "natural language processing", "text classification", "named entity recognition", "ner", "sentiment analysis", "text mining"],
    "Computer Vision":  ["computer vision", "cv", "image classification", "object detection", "yolo", "opencv", "image recognition"],
    "TensorFlow":       ["tensorflow", "tf", "keras"],
    "PyTorch":          ["pytorch", "torch"],
    "Scikit-learn":     ["scikit-learn", "sklearn", "scikit learn"],
    "Hugging Face":     ["hugging face", "huggingface", "transformers", "bert", "gpt", "llama"],
    "LangChain":        ["langchain", "lang chain", "llamaindex", "llama index"],
    "LLMs":             ["llm", "llms", "large language model", "gpt-4", "chatgpt", "prompt engineering", "rag", "retrieval augmented"],
    "MLflow":           ["mlflow", "ml flow", "model registry", "experiment tracking"],
    "Feature Engineering": ["feature engineering", "feature selection", "feature extraction"],
    "Reinforcement Learning": ["reinforcement learning", "rl", "q-learning", "dqn"],
    "Generative AI":    ["generative ai", "gen ai", "genai", "diffusion model", "stable diffusion", "gans", "vae"],
    "MLOps":            ["mlops", "ml ops", "model deployment", "model serving"],

    # ── Data Science & Analytics ─────────────────────────────────────────────
    "Data Analysis":    ["data analysis", "data analytics", "exploratory data analysis", "eda"],
    "Data Visualization": ["data visualization", "tableau", "power bi", "powerbi", "looker", "matplotlib", "seaborn", "plotly", "d3.js"],
    "Pandas":           ["pandas"],
    "NumPy":            ["numpy", "np"],
    "Statistics":       ["statistics", "statistical analysis", "hypothesis testing", "regression", "a/b testing"],
    "Spark":            ["spark", "apache spark", "pyspark", "databricks"],
    "Airflow":          ["airflow", "apache airflow", "dbt", "data pipeline"],
    "ETL":              ["etl", "data pipeline", "data ingestion", "data warehouse"],

    # ── Vector Databases & Search ────────────────────────────────────────────
    "Pinecone":         ["pinecone"],
    "ChromaDB":         ["chromadb", "chroma"],
    "Weaviate":         ["weaviate"],
    "FAISS":            ["faiss", "vector search", "vector store", "embedding search"],

    # ── Mobile Development ───────────────────────────────────────────────────
    "Android":          ["android", "android studio"],
    "iOS":              ["ios", "xcode"],
    "Flutter":          ["flutter", "dart"],
    "React Native":     ["react native"],

    # ── Software Engineering ─────────────────────────────────────────────────
    "Git":              ["git", "github", "gitlab", "bitbucket", "version control"],
    "Agile/Scrum":      ["agile", "scrum", "kanban", "sprint", "jira"],
    "System Design":    ["system design", "distributed systems", "microservices", "scalability"],
    "Design Patterns":  ["design patterns", "solid principles", "oop", "object oriented"],
    "Testing":          ["unit testing", "integration testing", "tdd", "pytest", "jest", "selenium", "cypress"],
    "Security":         ["cybersecurity", "owasp", "penetration testing", "ssl", "oauth", "jwt", "encryption"],

    # ── Project Management / Soft Skills ─────────────────────────────────────
    "Leadership":       ["leadership", "team lead", "tech lead", "managed team", "mentoring"],
    "Communication":    ["communication", "presentation skills", "stakeholder management"],
    "Problem Solving":  ["problem solving", "analytical thinking", "critical thinking"],

    # ── Domain-Specific ──────────────────────────────────────────────────────
    "FinTech":          ["fintech", "banking", "payments", "trading", "financial modeling"],
    "Healthcare AI":    ["healthcare", "medical imaging", "ehr", "fhir", "clinical"],
    "Blockchain":       ["blockchain", "solidity", "web3", "smart contracts", "ethereum"],
    "IoT":              ["iot", "internet of things", "embedded systems", "raspberry pi", "arduino"],

    # ── Robotics ─────────────────────────────────────────────────────────────
    "ROS":              ["ros", "ros1", "robot operating system", "roslaunch", "rostopic", "roscpp", "rospy", "rosnode"],
    "ROS2":             ["ros2", "ros 2", "ros-2", "rclpy", "rclcpp", "colcon", "ros humble", "ros galactic", "ros foxy"],
    "Gazebo":           ["gazebo", "gazebo simulator", "gazebo sim", "gzebo"],
    "MoveIt":           ["moveit", "moveit2", "move-it"],
    "SLAM":             ["slam", "simultaneous localization and mapping", "lidar slam", "visual slam", "cartographer", "gmapping", "hector slam", "rtab-map", "orb-slam"],
    "Path Planning":    ["path planning", "rrt", "rrt*", "prm", "a-star", "a* pathfinding", "dijkstra", "nav2", "navigation stack"],
    "Robot Kinematics": ["kinematics", "inverse kinematics", "forward kinematics", "denavit hartenberg", "dh parameters", "jacobian"],
    "Robot Manipulation": ["robot manipulation", "grasping", "pick and place", "robotic arm", "end effector", "gripper"],
    "Robot Perception": ["robot perception", "3d perception", "depth camera", "stereo vision", "lidar", "sonar", "rangefinder"],
    "Autonomous Navigation": ["autonomous navigation", "autonomous driving", "self driving", "indoor navigation", "amcl", "localization"],
    "Drone/UAV":        ["drone", "uav", "unmanned aerial", "quadrotor", "quadcopter", "ardupilot", "px4", "mavros", "mission planner", "qgroundcontrol"],
    "Control Systems":  ["control systems", "pid", "pid controller", "feedback control", "mpc", "model predictive control", "lqr", "state space control"],
    "Robot Simulation": ["webots", "coppeliasim", "v-rep", "isaac sim", "pybullet", "mujoco", "gazebo classic"],

    # ── Embedded Systems & Hardware ───────────────────────────────────────────
    "RTOS":             ["rtos", "real-time os", "real time operating system", "freertos", "zephyr rtos", "vxworks", "qnx", "threadx"],
    "FPGA":             ["fpga", "field programmable gate array", "xilinx", "altera", "intel fpga", "quartus", "vivado"],
    "Verilog/VHDL":     ["verilog", "vhdl", "system verilog", "hdl", "hardware description language"],
    "CUDA":             ["cuda", "gpu programming", "gpu computing", "gpu acceleration", "opencl", "cudnn", "cublas", "thrust"],
    "Embedded C":       ["embedded c", "bare metal", "microcontroller programming", "stm32", "avr", "pic microcontroller", "atmega", "arm cortex", "nrf52", "esp32", "esp8266"],
    "Communication Protocols": ["uart", "spi", "i2c", "can bus", "canopen", "rs232", "rs485", "modbus", "profibus", "ethercat", "profinet"],
    "PCB Design":       ["pcb", "pcb design", "kicad", "eagle cad", "altium designer", "circuit design", "schematic design"],
    "Signal Processing": ["signal processing", "dsp", "digital signal processing", "fft", "filter design", "kalman filter", "particle filter", "iir filter", "fir filter"],
    "MATLAB/Simulink":  ["simulink", "matlab simulink", "stateflow", "simscape", "embedded coder"],
    "Linux Embedded":   ["yocto", "buildroot", "embedded linux", "openwrt", "petalinux", "poky"],

    # ── Game Development ──────────────────────────────────────────────────────
    "Unity":            ["unity", "unity3d", "unity engine", "unity game engine"],
    "Unreal Engine":    ["unreal", "unreal engine", "ue4", "ue5", "blueprints", "unreal blueprints", "lumen", "nanite"],
    "Godot":            ["godot", "gdscript", "godot engine", "godot 4"],
    "OpenGL":           ["opengl", "glsl", "glfw", "glad", "open gl"],
    "Vulkan":           ["vulkan", "vulkan api", "vulkan sdk"],
    "DirectX":          ["directx", "direct3d", "hlsl", "dx11", "dx12", "directx 12"],
    "Shader Programming": ["shader", "shaders", "shader programming", "vertex shader", "fragment shader", "compute shader"],
    "Game Physics":     ["game physics", "physics engine", "box2d", "bullet physics", "physx", "havok"],

    # ── AR/VR & 3D ────────────────────────────────────────────────────────────
    "AR/VR":            ["ar", "vr", "augmented reality", "virtual reality", "mixed reality", "xr", "extended reality", "metaverse"],
    "WebXR":            ["webxr", "webvr", "aframe", "a-frame", "three.js", "threejs", "babylon.js"],
    "ARKit/ARCore":     ["arkit", "arcore", "ar foundation", "vuforia"],
    "3D Modeling":      ["3d modeling", "blender", "3ds max", "maya 3d", "cinema4d", "zbrush", "solidworks", "autocad", "cad"],
    "Point Cloud":      ["point cloud", "lidar processing", "open3d", "pcl", "pointnet", "pcd"],

    # ── Networking ────────────────────────────────────────────────────────────
    "Networking":       ["networking", "tcp/ip", "network protocols", "osi model", "dns", "dhcp", "subnetting", "routing"],
    "MQTT":             ["mqtt", "mqtt broker", "mosquitto", "hivemq", "emqx", "mqtt protocol"],
    "WebRTC":           ["webrtc", "web rtc", "real-time communication", "video streaming", "webrtc protocol"],
    "VPN/Firewall":     ["vpn", "proxy", "reverse proxy", "load balancing", "nginx proxy", "haproxy"],
    "Network Security": ["network security", "firewall", "wireshark", "packet analysis", "intrusion detection", "ids", "ips"],
    "5G/Wireless":      ["5g", "lte", "4g", "wifi", "bluetooth", "ble", "zigbee", "lorawan", "lora", "nfc", "rf", "wireless"],

    # ── Programming Languages (Extended) ─────────────────────────────────────
    "Julia":            ["julia", "julia lang", "julia programming"],
    "Haskell":          ["haskell"],
    "Erlang":           ["erlang"],
    "Elixir":           ["elixir", "phoenix framework"],
    "Perl":             ["perl", "perl scripting"],
    "Lua":              ["lua", "lua scripting"],
    "Assembly":         ["assembly", "asm", "x86 assembly", "arm assembly", "mips assembly"],
    "Prolog":           ["prolog", "logic programming", "prolog programming"],
    "Fortran":          ["fortran", "fortran 90", "fortran 77"],
    "COBOL":            ["cobol", "cobol programming"],
    "Groovy":           ["groovy"],
    "F#":               ["f#", "fsharp"],
    "OCaml":            ["ocaml"],
    "Zig":              ["zig programming"],
    "Nim":              ["nim lang", "nim programming"],

    # ── Databases (Extended) ──────────────────────────────────────────────────
    "Neo4j":            ["neo4j", "graph database", "cypher query", "arangodb", "janusgraph"],
    "InfluxDB":         ["influxdb", "time series database", "tsdb", "timescaledb"],
    "Oracle DB":        ["oracle", "oracle database", "oracle sql"],
    "MS SQL Server":    ["sql server", "mssql", "microsoft sql server"],
    "Firebase":         ["firebase", "firestore", "firebase realtime database"],
    "Supabase":         ["supabase"],
    "CouchDB":          ["couchdb", "pouchdb"],
    "HBase":            ["hbase", "apache hbase", "hadoop hbase"],
    "MariaDB":          ["mariadb"],

    # ── Design & UI/UX ────────────────────────────────────────────────────────
    "Figma":            ["figma"],
    "Adobe XD":         ["adobe xd"],
    "Photoshop":        ["photoshop", "adobe photoshop"],
    "Illustrator":      ["illustrator", "adobe illustrator"],
    "UI/UX Design":     ["ui design", "ux design", "ui/ux", "user interface design", "user experience design", "wireframing", "prototyping"],
    "Sketch":           ["sketch app", "sketch design"],

    # ── Cloud (Extended) ──────────────────────────────────────────────────────
    "Serverless":       ["serverless", "aws lambda", "azure functions", "cloud functions", "faas", "function as a service"],
    "Cloudflare":       ["cloudflare", "cloudflare workers", "cloudflare cdn"],
    "Vercel":           ["vercel"],
    "Heroku":           ["heroku"],
    "DigitalOcean":     ["digitalocean", "digital ocean"],

    # ── Scientific Computing ──────────────────────────────────────────────────
    "SciPy":            ["scipy", "scientific python", "scientific computing"],
    "OpenCV":           ["opencv", "cv2", "open cv"],
    "Jupyter":          ["jupyter", "jupyter notebook", "jupyter lab", "ipynb"],
    "SymPy":            ["sympy", "symbolic math", "symbolic computation"],
    "Google Colab":     ["colab", "google colab", "colaboratory"],

    # ── Bioinformatics ────────────────────────────────────────────────────────
    "Bioinformatics":   ["bioinformatics", "genomics", "proteomics", "biopython", "blast", "sequence alignment", "bioconductor"],

    # ── Optimization ─────────────────────────────────────────────────────────
    "Optimization":     ["optimization", "linear programming", "integer programming", "convex optimization", "gurobi", "cplex", "or-tools", "scipy optimize"],

    # ── ROS Middleware / DDS ──────────────────────────────────────────────────
    "DDS":              ["dds", "fast dds", "cyclone dds", "zenoh", "micro-ros", "micro ros", "data distribution service"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SKILL INFERENCE RULES
# If a candidate has skill A, we can infer skill B
# ─────────────────────────────────────────────────────────────────────────────
INFERENCE_RULES = {
    "Deep Learning":        ["TensorFlow", "PyTorch", "Keras"],
    "Machine Learning":     ["TensorFlow", "PyTorch", "Scikit-learn", "Deep Learning"],
    "NLP":                  ["Hugging Face", "LLMs", "LangChain", "spaCy"],
    "MLOps":                ["Docker", "Kubernetes", "MLflow"],
    "LLM Engineering":      ["LangChain", "LLMs", "Hugging Face"],
    "REST API":             ["FastAPI", "Django", "Flask", "Node.js", "Spring Boot"],
    "Data Engineering":     ["Spark", "Airflow", "ETL", "Kafka"],
    "Full Stack":           ["React", "Node.js", "PostgreSQL"],
    "Vector Search":        ["FAISS", "Pinecone", "ChromaDB", "Weaviate"],
    "Python":               ["Django", "Flask", "FastAPI", "PyTorch", "TensorFlow"],
}

# ─────────────────────────────────────────────────────────────────────────────
# ROLE SUGGESTION RULES
# ─────────────────────────────────────────────────────────────────────────────
ROLE_RULES = [
    {
        "role": "ML Engineer",
        "required": ["Machine Learning"],
        "bonus": ["TensorFlow", "PyTorch", "MLOps", "Docker", "Python"]
    },
    {
        "role": "LLM / GenAI Engineer",
        "required": ["LLMs"],
        "bonus": ["LangChain", "Hugging Face", "Generative AI", "Python", "RAG"]
    },
    {
        "role": "Data Scientist",
        "required": ["Data Analysis"],
        "bonus": ["Statistics", "Machine Learning", "Python", "Pandas", "Spark"]
    },
    {
        "role": "Backend Developer",
        "required": ["REST API"],
        "bonus": ["Python", "Node.js", "Java", "PostgreSQL", "Docker"]
    },
    {
        "role": "Frontend Developer",
        "required": ["React"],
        "bonus": ["JavaScript", "TypeScript", "Vue.js", "HTML/CSS", "Next.js"]
    },
    {
        "role": "Full Stack Developer",
        "required": ["Full Stack"],
        "bonus": ["JavaScript", "React", "Node.js", "PostgreSQL", "Docker"]
    },
    {
        "role": "DevOps / Cloud Engineer",
        "required": ["Docker"],
        "bonus": ["Kubernetes", "AWS", "Terraform", "CI/CD", "Linux"]
    },
    {
        "role": "Data Engineer",
        "required": ["ETL"],
        "bonus": ["Spark", "Airflow", "SQL", "Python", "Kafka", "Snowflake"]
    },
    {
        "role": "Android Developer",
        "required": ["Android"],
        "bonus": ["Kotlin", "Java", "Firebase", "REST API"]
    },
    {
        "role": "iOS Developer",
        "required": ["iOS"],
        "bonus": ["Swift", "SwiftUI", "Xcode"]
    },
    {
        "role": "Cybersecurity Engineer",
        "required": ["Security"],
        "bonus": ["Linux", "Networking", "Python", "Cloud"]
    },
    {
        "role": "Blockchain Developer",
        "required": ["Blockchain"],
        "bonus": ["Solidity", "Web3", "Smart Contracts", "Python"]
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def extract_skills(text: str) -> list[str]:
    """
    Extract canonical skill names from free-form text using alias matching.
    Returns list of canonical skill names (e.g., "Machine Learning", "Docker").
    """
    text_lower = text.lower()
    # Add spaces around text to avoid partial-word matches for short skills
    padded = f" {text_lower} "
    found = set()

    for canonical, aliases in SKILLS_TAXONOMY.items():
        for alias in aliases:
            # Use word-boundary matching for short tokens (e.g. "r", "go")
            pattern = r'(?<![a-z0-9])' + re.escape(alias) + r'(?![a-z0-9])'
            if re.search(pattern, padded):
                found.add(canonical)
                break  # No need to check more aliases for this skill

    return list(found)


def infer_skills(known_skills: list[str]) -> list[str]:
    """
    Given a list of known skills, infer additional implied skills.
    E.g., if candidate has TensorFlow + PyTorch → infer "Deep Learning".
    """
    known_set = set(known_skills)
    inferred = set()

    for inferred_skill, sources in INFERENCE_RULES.items():
        # Infer if at least 1 source skill is known and inferred skill isn't already
        if any(s in known_set for s in sources) and inferred_skill not in known_set:
            inferred.add(inferred_skill)

    return list(inferred)


def compare_skills(resume_skills: list[str], jd_skills: list[str]) -> tuple[list, list]:
    """
    Compare resume skills vs job description skills.
    Returns (matched_list, missing_list).
    """
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)
    matched = list(resume_set & jd_set)
    missing = list(jd_set - resume_set)
    return matched, missing


def suggest_roles(skills: list[str]) -> list[str]:
    """
    Suggest job roles based on candidate's skill set using rule-based matching.
    Returns up to 3 most relevant roles.
    """
    skill_set = set(skills)
    scored_roles = []

    for rule in ROLE_RULES:
        # Must have at least one required skill
        if any(req in skill_set for req in rule["required"]):
            score = sum(1 for b in rule["bonus"] if b in skill_set)
            scored_roles.append((rule["role"], score))

    # Sort by match score descending
    scored_roles.sort(key=lambda x: -x[1])
    return [r[0] for r in scored_roles[:3]] or ["Software Engineer", "Technical Analyst"]