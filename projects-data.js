// projects-data.js - Comprehensive database of Shiva Kumar Madagoni's projects
const PROJECTS_DATA = [
  {
    id: "orchest-ai",
    title: "OrchestAI — AI Workforce Orchestration Platform",
    repo: "Employee_Work_Management",
    badge: "Enterprise SaaS",
    category: "agentic",
    featured: true,
    tagline: "Next-gen enterprise platform orchestrating employees, tasks, and DAG workflows with AI",
    summary: "Full-stack enterprise solution combining FastAPI, Next.js 16, PostgreSQL, and LLMs to intelligently manage employees and assign projects using DAG dependency graphs and real-time WebSockets.",
    metrics: "Automates squad matching across 50+ employee skill sets with sub-100ms DAG updates",
    stack: ["FastAPI", "Next.js 16", "PostgreSQL", "React Flow", "WebSockets", "Gemini AI", "SQLAlchemy"],
    githubUrl: "https://github.com/ShivaKumar338/Employee_Work_Management",
    liveUrl: null,
    highlights: [
      "Built interactive DAG (Directed Acyclic Graph) workflow visualization using React Flow and custom dependency locks.",
      "Engineered automated skill-matrix matching algorithm to rank and recommend optimal employee squads.",
      "Implemented real-time bi-directional task status synchronization via WebSockets.",
      "Constructed secure role-based access control (RBAC) with JWT auth, refresh tokens, and PostgreSQL schema."
    ],
    architecture: "Frontend (Next.js 16 App Router + TanStack Query) <-> REST & WebSocket <-> FastAPI Backend <-> PostgreSQL + AI Engine"
  },
  {
    id: "deepguard-ai",
    title: "DeepGuard AI — Multimodal Deepfake Detection Agent",
    repo: "Deep-fake-Ai-detection",
    badge: "Autonomous AI",
    category: "vision",
    featured: true,
    tagline: "Autonomous multimodal agent identifying manipulated media across image, video, audio, and text",
    summary: "An autonomous agent system built on FastAPI and React for intelligent deepfake detection across four modalities with RAG-powered forensic explanations and memory-based analysis history.",
    metrics: "Supports 4 modalities with automated media type classification and deep learning inference",
    stack: ["FastAPI", "React", "TensorFlow / Keras", "OpenCV", "RAG", "TailwindCSS", "Python"],
    githubUrl: "https://github.com/ShivaKumar338/Deep-fake-Ai-detection",
    liveUrl: null,
    highlights: [
      "Engineered autonomous media classification engine that auto-detects incoming media types without manual user selection.",
      "Integrated deep learning model inference (.h5 pipelines) specialized for facial manipulation and voice spoofing detection.",
      "Designed RAG-powered forensic explanation module breaking down detection confidence scores into human-readable rationale.",
      "Implemented session memory to track multi-step analysis history and recurrent media verification."
    ],
    architecture: "React Multimodal UI -> FastAPI Inference Service -> Pre-trained Neural Models (.h5) -> Forensic RAG Explanation Engine"
  },
  {
    id: "tech-routing-ai",
    title: "AI Technician Routing & Automated Negotiation System",
    repo: "AI-Technician-Selection-Routing-Automated-Negotiation-System",
    badge: "Multi-Agent System",
    category: "agentic",
    featured: true,
    tagline: "Geospatial routing score and twin Gemini agent automated price negotiation",
    summary: "FastAPI and Supabase backend that matches customer service requests to technicians using a weighted multi-factor routing score, then autonomously negotiates pricing using two conversational Gemini agents.",
    metrics: "Twin-agent autonomous negotiation converging contracts within 3-5 rounds",
    stack: ["FastAPI", "Gemini 1.5 Flash", "Supabase", "PostgreSQL", "Haversine Distance", "Python"],
    githubUrl: "https://github.com/ShivaKumar338/AI-Technician-Selection-Routing-Automated-Negotiation-System",
    liveUrl: null,
    highlights: [
      "Formulated geospatial routing algorithm combining Haversine proximity calculations with skill-matrix matching.",
      "Created dual autonomous LLM negotiation loop: Customer Agent vs Technician Agent with budget and margin boundaries.",
      "Developed Supabase PostgreSQL schema storing technician profiles, dynamic job statuses, and live negotiation audit logs.",
      "Documented clean RESTful endpoints: /seed, /jobs, /match, and /negotiate with OpenAPI schemas."
    ],
    architecture: "Customer Request -> Weighted Haversine & Skill Ranker -> Twin Gemini Agents (Customer vs Tech) -> Supabase Transaction DB"
  },
  {
    id: "vision-ai",
    title: "VisionAI — Voice & Vision Edge Assistant",
    repo: "Vision-Ai-Assistance",
    badge: "Computer Vision",
    category: "vision",
    featured: true,
    tagline: "Real-time edge assistant combining YOLOv8 object detection, TrOCR, and interactive voice control",
    summary: "Comprehensive edge vision and audio assistant combining real-time YOLOv8 object localization, EasyOCR for printed text, Hugging Face TrOCR for handwritten notes, and full two-way voice conversation.",
    metrics: "Sub-second OCR & handwriting recognition with real-time text-to-speech feedback",
    stack: ["Python", "OpenCV", "YOLOv8", "Hugging Face TrOCR", "EasyOCR", "pyttsx3", "SpeechRecognition"],
    githubUrl: "https://github.com/ShivaKumar338/Vision-Ai-Assistance",
    liveUrl: null,
    highlights: [
      "Integrated YOLOv8 neural network for high-fps multi-class object detection and tracking via webcam.",
      "Implemented dual OCR pipeline: EasyOCR for printed documents and Vision Transformer (TrOCR) for handwritten text.",
      "Built voice command listener (SpeechRecognition) enabling hands-free queries like 'read handwriting' or 'read notes'.",
      "Constructed offline voice synthesis engine with pyttsx3 and persistent scratchpad logging in visionai_notes.txt."
    ],
    architecture: "Webcam / Audio Input -> SpeechRecognition -> YOLOv8 / TrOCR Pipeline -> pyttsx3 Voice Synthesis Engine"
  },
  {
    id: "flow-intel",
    title: "FlowIntel — AI Workflow Intelligence Platform",
    repo: "workflow-intelligence-platform",
    badge: "Enterprise Monorepo",
    category: "fullstack",
    featured: false,
    tagline: "Automated PRD extraction, dependency-locked DAGs, and real-time team dispatch",
    summary: "Monorepo with Next.js App Router, FastAPI, PostgreSQL, Redis/Celery, and Gemini AI. Transforms unstructured requirements into task breakdowns, ranked squads, and real-time DAG stages.",
    metrics: "Async Celery queue processing with live WebSocket status streaming",
    stack: ["Next.js", "FastAPI", "SQLAlchemy Async", "PostgreSQL", "Celery", "Redis", "Gemini AI"],
    githubUrl: "https://github.com/ShivaKumar338/workflow-intelligence-platform",
    liveUrl: null,
    highlights: [
      "Leveraged Gemini LLM for automated PRD synthesis and decomposing business goals into dependency stages.",
      "Architected async DAG engine with SQLAlchemy models handling edge dependencies, approvals, and reject logic.",
      "Configured Celery background workers with Redis broker for asynchronous email notifications and analytics calculation.",
      "Built interactive manager dashboard and employee productivity analytics with WebSocket live updates."
    ],
    architecture: "Next.js UI -> FastAPI Async -> Redis / Celery Workers + PostgreSQL -> WebSocket Live Stream"
  },
  {
    id: "crew-ai-pipeline",
    title: "CrewAI — Multi-Agent Market Intelligence Pipeline",
    repo: "CREW_AI",
    badge: "Multi-Agent System",
    category: "agentic",
    featured: false,
    tagline: "Autonomous 3-agent pipeline scraping, cross-matching, and synthesizing market data",
    summary: "Automated multi-agent pipeline orchestrating 3 specialized autonomous agents (DataCollector, ProductMatcher, and DataFormatter) powered by Google Gemini to scrape, match, and summarize market data into CSV.",
    metrics: "3-agent collaborative pipeline with autonomous data reconciliation",
    stack: ["Python", "CrewAI Flow", "Gemini 1.5 Flash", "Web Scraping", "Pandas", "ETL"],
    githubUrl: "https://github.com/ShivaKumar338/CREW_AI",
    liveUrl: null,
    highlights: [
      "Built DataCollector agent for web scraping prediction sites and aggregating unstandardized product listings.",
      "Engineered ProductMatcher agent using LLM semantic reasoning to merge duplicate items from heterogeneous sources.",
      "Formulated DataFormatter agent converting merged data into structured CSV files and executive market summaries.",
      "Implemented modular class structure with secure environment credential handling."
    ],
    architecture: "Scraper Source -> DataCollector Agent -> ProductMatcher Agent -> DataFormatter Agent -> CSV & Executive Summary"
  },
  {
    id: "fraud-detection",
    title: "Financial Fraud Detection System (Live on Render)",
    repo: "Fraud_Detection",
    badge: "Live Deployment",
    category: "ml",
    featured: true,
    tagline: "Production ML model identifying fraudulent transactions, deployed with FastAPI on Render",
    summary: "Machine learning based transaction classification system analyzing high-volume credit card records to accurately flag suspicious activities. Packaged with a production FastAPI backend and deployed live.",
    metrics: "Live production deployment on Render handling real-time prediction requests",
    stack: ["Python", "Scikit-learn", "FastAPI", "Pandas", "NumPy", "Matplotlib", "Render"],
    githubUrl: "https://github.com/ShivaKumar338/Fraud_Detection",
    liveUrl: "https://fraud-detection-dnty.onrender.com",
    highlights: [
      "Trained classification algorithms (Random Forest, Logistic Regression) on imbalanced financial fraud datasets.",
      "Applied SMOTE and advanced feature scaling to maximize ROC-AUC and minimize false positive disruptions.",
      "Encapsulated prediction pipeline in a low-latency FastAPI REST endpoint.",
      "Deployed on Render cloud infrastructure using Uvicorn and Gunicorn."
    ],
    architecture: "Transaction Record JSON -> FastAPI REST API -> Scikit-learn Inference Engine -> Fraud Probability Classification"
  },
  {
    id: "sakha-ai",
    title: "SAKHA-AI — Generative AI Conversational Assistant",
    repo: "SAKHA-AI",
    badge: "Generative AI",
    category: "agentic",
    featured: false,
    tagline: "Lightweight conversational web assistant powered by Google Gemini models",
    summary: "Web-based interactive chatbot built with Flask and Google Gemini AI, offering smooth natural language interaction, prompt conditioning, and clean web interface.",
    metrics: "Real-time query processing powered by Google's flagship multimodal LLM",
    stack: ["Python", "Flask", "Google Gemini API", "JavaScript", "HTML5", "CSS3"],
    githubUrl: "https://github.com/ShivaKumar338/SAKHA-AI",
    liveUrl: null,
    highlights: [
      "Developed backend conversational endpoint using Flask connecting directly to Google Gemini API.",
      "Engineered system prompts for consistent contextual awareness and helpful tone.",
      "Created sleek browser client with typing indicators, responsive layout, and error resilience.",
      "Demonstrated practical full-stack Generative AI integration patterns."
    ],
    architecture: "Browser Chat UI -> Flask Server -> Google Gemini API -> Formatted Markdown Stream"
  },
  {
    id: "predictive-maintenance",
    title: "Machine Failure Predictive Maintenance System",
    repo: "Machine-Failure-Prediction-Project",
    badge: "Industrial IoT ML",
    category: "ml",
    featured: false,
    tagline: "Predictive maintenance framework forecasting mechanical breakdown to cut downtime",
    summary: "Predictive maintenance machine learning project analyzing operational sensor data (vibration, heat dissipation, load, torque) to predict equipment failure before it occurs.",
    metrics: "Reduced simulated industrial downtime by ~15% via early failure detection",
    stack: ["Python", "Jupyter Notebook", "Scikit-learn", "Pandas", "Seaborn", "IoT Data"],
    githubUrl: "https://github.com/ShivaKumar338/Machine-Failure-Prediction-Project",
    liveUrl: null,
    highlights: [
      "Conducted extensive exploratory data analysis (EDA) across multi-sensor telemetry records.",
      "Modeled thermal and mechanical stress indicators to identify pre-failure warning signals.",
      "Benchmarked multiple machine learning classifiers to balance precision and recall on catastrophic failures.",
      "Formulated actionable maintenance scheduling metrics resulting in ~15% reduction in unplanned downtime."
    ],
    architecture: "IoT Sensor Streams -> Preprocessing & Feature Engineering -> Scikit-learn Model -> Health Score & Alert Thresholds"
  },
  {
    id: "agent-x",
    title: "AgentX / AI-Buddy — Configurable Memory Agent Starter",
    repo: "AI-Buddy",
    badge: "Agent Template",
    category: "agentic",
    featured: false,
    tagline: "Production-ready open-source template for memory-powered AI chat agents",
    summary: "Modular conversational agent starter kit featuring short-term memory, personality customization via single config file (agent.config.js), and zero-boilerplate deployment.",
    metrics: "5-minute zero-code agent customization via unified configuration schema",
    stack: ["JavaScript", "React", "Gemini API", "Vercel", "TailwindCSS"],
    githubUrl: "https://github.com/ShivaKumar338/AI-Buddy",
    liveUrl: null,
    highlights: [
      "Architected clean configuration pattern allowing agent personality and tool tuning in agent.config.js.",
      "Integrated Google AI Studio Gemini API with stateful conversation memory management.",
      "Designed responsive UI with dynamic starter topic chips and markdown rendering."
    ],
    architecture: "Unified Config -> Context Memory Manager -> Gemini API -> Interactive Client"
  },
  {
    id: "neetcode-dsa",
    title: "Algorithmic Problem Solving & NeetCode 150 Archive",
    repo: "neetcode-submissions",
    badge: "Algorithms & CS",
    category: "fullstack",
    featured: false,
    tagline: "Rigorous algorithmic implementations with optimal time and space complexity",
    summary: "Repository of solutions to core Data Structures & Algorithms challenges including dynamic programming, graph traversals, binary search trees, and sliding window paradigms.",
    metrics: "100+ algorithmic problems solved with rigorous time/space complexity analysis",
    stack: ["Python", "Algorithms", "Data Structures", "LeetCode / NeetCode"],
    githubUrl: "https://github.com/ShivaKumar338/neetcode-submissions",
    liveUrl: "https://leetcode.com/u/shivakumar338/",
    highlights: [
      "Systematic implementations covering Arrays & Hashing, Two Pointers, Trees, Backtracking, and DP.",
      "Focused on writing clean, idiomatic, and interview-ready Pythonic solutions.",
      "Benchmark test cases validating edge condition performance and memory constraints."
    ],
    architecture: "Problem Specification -> Mathematical Proof / Logic -> Python Implementation -> Asymptotic Analysis"
  },
  {
    id: "hackathon-generator",
    title: "Hackathon Idea Generator REST API",
    repo: "Hackathon-Idea-Generator",
    badge: "Web Service",
    category: "fullstack",
    featured: false,
    tagline: "Flask + SQLite API providing instant thematic brainstorm suggestions",
    summary: "Lightweight REST API serving categorized problem statements across AI, FinTech, Healthcare, and Cybersecurity to accelerate hackathon ideation and development.",
    metrics: "Sub-10ms response time with SQLite indexing and category filtering",
    stack: ["Python", "Flask", "SQLite3", "REST API", "JSON"],
    githubUrl: "https://github.com/ShivaKumar338/Hackathon-Idea-Generator",
    liveUrl: null,
    highlights: [
      "Developed structured SQLite schema storing domain-specific problem statements and project prompts.",
      "Exposed clean parameterized GET endpoints (/get_idea?category=...) with JSON payload validation.",
      "Built resilient error handling for fallback categorization and randomized discovery."
    ],
    architecture: "HTTP Client -> Flask Request Router -> SQLite Query Engine -> JSON Response"
  }
];
