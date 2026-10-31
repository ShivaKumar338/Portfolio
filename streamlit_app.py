import streamlit as st
import pandas as pd
import numpy as np
import base64
import os

# Configure Page
st.set_page_config(
    page_title="Shiva Kumar Madagoni | AI Engineer & Full-Stack Portfolio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dark UI CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .gradient-header {
        background: linear-gradient(135deg, #06b6d4 0%, #6366f1 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }

    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .project-card {
        background: rgba(16, 23, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 22px;
        transition: all 0.3s ease;
    }
    .project-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.2);
    }

    .metric-badge {
        background: rgba(6, 182, 212, 0.12);
        color: #22d3ee;
        border-left: 3px solid #06b6d4;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 10px 0;
    }

    .tech-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #cbd5e1;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .stat-box {
        background: rgba(16, 23, 38, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .stat-val {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #06b6d4;
    }
    .stat-desc {
        font-size: 0.85rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get image base64
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# Sidebar Navigation
img_b64 = get_image_base64("Shiva.png") or get_image_base64("Portfolio/Shiva.png")

with st.sidebar:
    if img_b64:
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 15px;">
                <img src="data:image/png;base64,{img_b64}" 
                     style="width: 140px; height: 140px; border-radius: 50%; object-fit: cover; border: 3px solid #6366f1; box-shadow: 0 8px 20px rgba(0,0,0,0.5);">
                <h3 style="margin-top: 10px; margin-bottom: 2px;">Shiva Kumar Madagoni</h3>
                <p style="color: #06b6d4; font-size: 0.85rem; font-weight: 600;">AI Engineer | Full-Stack</p>
                <p style="color: #94a3b8; font-size: 0.75rem;">Pallavi Engineering College</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🧭 Navigation")
    selection = st.radio(
        "Explore Section:",
        ["🚀 Home & Bio", "💼 Experience", "🛠️ Featured Projects", "📊 Skills & Tech Stack", "📫 Contact & Connect"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown(
        """
        - [🐙 **GitHub Profile**](https://github.com/ShivaKumar338)
        - [💼 **LinkedIn Profile**](https://www.linkedin.com/in/shiva-kumar-madagoni-627038327/)
        - [📧 **Direct Email**](mailto:mshivakumar1289@gmail.com)
        - [🌐 **Interactive Web UI**](http://localhost:8000) *(Open `index.html`)*
        """
    )
    st.caption("© 2026 Shiva Kumar Madagoni • Open for Internships")

# ==========================================
# Section 1: Home & Bio
# ==========================================
if selection == "🚀 Home & Bio":
    st.markdown('<div class="status-chip">🟢 Available for Internship</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="gradient-header">Shiva Kumar Madagoni</h1>', unsafe_allow_html=True)
    st.markdown("#### **Artificial Intelligence & Machine Learning Student | Agentic AI & Full-Stack Systems Developer**")

    st.markdown(
        """
        I am an undergraduate in **Artificial Intelligence and Machine Learning (2024 - 2028)** at **Pallavi Engineering College**, Greater Hyderabad Area.
        I specialize in engineering **Autonomous Multi-Agent Workflows** (CrewAI, Gemini), **Multimodal Deep Learning Models** (DeepGuard AI), **Computer Vision Systems** (YOLOv8, TrOCR), and scalable full-stack platforms using **FastAPI**, **Next.js**, and **PostgreSQL**.
        """
    )

    # Key Stats Strip
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="stat-box"><div class="stat-val">20+</div><div class="stat-desc">GitHub Repositories</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-box"><div class="stat-val">2</div><div class="stat-desc">AI Industry Traineeships</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-box"><div class="stat-val">4</div><div class="stat-desc">Modalities in DeepGuard AI</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-box"><div class="stat-val">100+</div><div class="stat-desc">DSA & NeetCode Solutions</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    col_about, col_edu = st.columns([1.1, 0.9])
    with col_about:
        st.subheader("💡 Engineering Philosophy")
        st.markdown(
            """
            My learning and engineering approach is **research-oriented and theory-backed**. I ground my implementations in literature such as *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* to deeply understand neural weights, convergence dynamics, and loss landscape optimization.

            - **Agentic AI:** Developing multi-agent autonomous systems capable of web scraping, automated negotiation, and dynamic PRD task generation.
            - **Multimodal Vision:** Real-time edge detection with YOLOv8, OCR on printed documents, and TrOCR handwriting deciphering.
            - **Robust APIs:** Production-grade RESTful backends and WebSocket feeds using FastAPI and PostgreSQL.
            """
        )

    with col_edu:
        st.subheader("🎓 Education")
        st.markdown(
            """
            **Bachelor of Technology (B.Tech) in Artificial Intelligence & Machine Learning**  
            *Pallavi Engineering College — Greater Hyderabad Area, India*  
            *Expected Graduation: 2028*

            **Key Academic Coursework:**  
            `Deep Learning` • `Data Structures & Algorithms` • `Computer Vision` • `PostgreSQL Database Design` • `Statistical Learning` • `Object Oriented Programming`
            """
        )
        st.info("💡 Regularly solving algorithmic challenges on NeetCode and LeetCode in Python with optimal asymptotic complexity.")

# ==========================================
# Section 2: Experience
# ==========================================
elif selection == "💼 Experience":
    st.header("💼 Industry Experience & Traineeships")
    st.markdown("Real-world technical experience in model training, deep learning frameworks, and predictive analytics.")

    # EdiGlobe
    with st.container():
        st.markdown(
            """
            ### 🏛️ Internship Trainee — Machine Learning & Neural Networks  
            **EdiGlobe (An IIT Kharagpur Initiative)** | *October 2025 – December 2025*  
            - Designed and implemented neural network pipelines utilizing **PyTorch** and **TensorFlow** for supervised learning and complex dataset interpretation.
            - Benchmarked and tuned multiple machine learning model families: Deep Neural Networks, Random Forests, Decision Trees, and Gradient Boosting.
            - Optimized data normalization, feature engineering, and loss convergence routines to improve business process predictions.
            - Strengthened deep learning fundamentals, model diagnostics, and hyperparameter optimization.
            
            *Technologies:* `PyTorch` • `TensorFlow` • `Neural Networks` • `Supervised Learning` • `Python` • `Data Optimization`
            """
        )
        st.markdown("---")

    # SkillDuniya
    with st.container():
        st.markdown(
            """
            ### 🤖 Training AI Intern  
            **SkillDuniya** | *March 2025 – May 2025*  
            - Engineered machine learning models for predictive analytics, classification, and workflow automation.
            - Utilized **Scikit-Learn, NumPy, and Pandas** to build reproducible data preprocessing and feature scaling pipelines.
            - Evaluated models using precision, recall, ROC-AUC, and cross-validation for enterprise datasets.
            - Acquired end-to-end perspective on the AI development lifecycle from EDA to deployment.
            
            *Technologies:* `Python` • `Scikit-Learn` • `Predictive Analytics` • `Pandas` • `NumPy` • `EDA`
            """
        )

# ==========================================
# Section 3: Featured Projects
# ==========================================
elif selection == "🛠️ Featured Projects":
    st.header("🛠️ Projects & Technical Builds")
    st.markdown("All repositories are public and actively maintained on [GitHub](https://github.com/ShivaKumar338).")

    proj_category = st.selectbox(
        "Filter by Domain:",
        ["All Projects", "Autonomous & Agentic AI", "Computer Vision & Multimodal", "Predictive ML & Data Science", "Full-Stack & Systems"]
    )

    # 1. OrchestAI
    if proj_category in ["All Projects", "Autonomous & Agentic AI", "Full-Stack & Systems"]:
        st.markdown(
            """
            ### 1. OrchestAI — AI Workforce Orchestration Platform [🔗](https://github.com/ShivaKumar338/Employee_Work_Management)
            **Enterprise SaaS Monorepo** | `FastAPI` `Next.js 16` `PostgreSQL` `React Flow` `WebSockets` `Gemini AI`
            - Full-stack enterprise platform that combines FastAPI, Next.js 16, PostgreSQL, and LLMs to intelligently allocate employees and manage tasks.
            - Features interactive **DAG (Directed Acyclic Graph)** dependency visualization using **React Flow**.
            - Automated skill-matrix matching algorithm recommending optimal employee squads.
            - Live task state synchronization powered by WebSockets.
            - Role-Based Access Control (RBAC) with JWT auth and refresh token rotation.
            """
        )
        st.markdown('<div class="metric-badge">Impact: Sub-100ms DAG updates and automated squad matching across 50+ employee skill sets</div>', unsafe_allow_html=True)
        st.markdown("---")

    # 2. DeepGuard AI
    if proj_category in ["All Projects", "Autonomous & Agentic AI", "Computer Vision & Multimodal"]:
        st.markdown(
            """
            ### 2. DeepGuard AI — Multimodal Deepfake Detection Agent [🔗](https://github.com/ShivaKumar338/Deep-fake-Ai-detection)
            **Autonomous AI System** | `FastAPI` `React` `TailwindCSS` `TensorFlow / Keras` `RAG` `Multimodal AI`
            - Autonomous agent system detecting manipulated media across **4 modalities: Image, Video, Audio, and Text**.
            - Autonomous media classification engine that auto-identifies inputs without manual user tagging.
            - Deep learning inference pipeline utilizing pre-trained `.h5` models.
            - **RAG-powered forensic explanations** that articulate detection confidence into human-understandable evidence.
            - Session memory tracking multi-step verification history.
            """
        )
        st.markdown('<div class="metric-badge">Impact: Unified multimodal agent architecture with RAG-backed forensic reporting</div>', unsafe_allow_html=True)
        st.markdown("---")

    # 3. AI Technician Routing
    if proj_category in ["All Projects", "Autonomous & Agentic AI"]:
        st.markdown(
            """
            ### 3. AI Technician Routing & Automated Negotiation API [🔗](https://github.com/ShivaKumar338/AI-Technician-Selection-Routing-Automated-Negotiation-System)
            **Multi-Agent Negotiation System** | `FastAPI` `Gemini 1.5 Flash` `Supabase` `Haversine Distance` `PostgreSQL`
            - Matches customer service requests to technicians using a weighted multi-factor scoring algorithm (skill-match + Haversine distance proximity).
            - Simulates automated price negotiation between two autonomous conversational Gemini agents (Customer Agent vs. Technician Agent).
            - Stores technician profiles, dynamic job statuses, and live negotiation logs in Supabase PostgreSQL.
            """
        )
        st.markdown('<div class="metric-badge">Impact: Twin-agent negotiation convergence in 3-5 rounds within defined margin boundaries</div>', unsafe_allow_html=True)
        st.markdown("---")

    # 4. VisionAI
    if proj_category in ["All Projects", "Computer Vision & Multimodal"]:
        st.markdown(
            """
            ### 4. VisionAI — AI Voice & Vision Edge Assistant [🔗](https://github.com/ShivaKumar338/Vision-Ai-Assistance)
            **Computer Vision & Edge AI** | `Python` `OpenCV` `YOLOv8` `TrOCR` `EasyOCR` `pyttsx3` `SpeechRecognition`
            - Real-time multimodal assistant combining **YOLOv8 object detection**, printed text reading (EasyOCR), and handwritten notes recognition (**Hugging Face TrOCR**).
            - Natural voice command control (`"read handwriting"`, `"read notes"`, `"time"`, `"date"`) with pyttsx3 text-to-speech audio responses.
            - Local notes persistence system with offline voice playback.
            """
        )
        st.markdown('<div class="metric-badge">Impact: Sub-second handwriting recognition and high-fps real-time object detection</div>', unsafe_allow_html=True)
        st.markdown("---")

    # 5. CrewAI
    if proj_category in ["All Projects", "Autonomous & Agentic AI"]:
        st.markdown(
            """
            ### 5. CrewAI — Multi-Agent Market Intelligence Pipeline [🔗](https://github.com/ShivaKumar338/CREW_AI)
            **Multi-Agent Flow** | `Python` `CrewAI Flow` `Gemini 1.5 Flash` `Web Scraping` `ETL`
            - Automated 3-agent pipeline: **DataCollector** (scrapes web listings) ➔ **ProductMatcher** (LLM semantic deduplication) ➔ **DataFormatter** (CSV export and executive market summary).
            - Orchestrated using CrewAI Flow pattern with Google Gemini integration.
            """
        )
        st.markdown("---")

    # 6. Fraud Detection
    if proj_category in ["All Projects", "Predictive ML & Data Science"]:
        st.markdown(
            """
            ### 6. Financial Fraud Detection System (Live on Render) [🔗](https://github.com/ShivaKumar338/Fraud_Detection)
            **Live Production Deployment** | `Scikit-Learn` `FastAPI` `Pandas` `NumPy` `Render`  
            👉 **[Open Live Demo on Render](https://fraud-detection-dnty.onrender.com)**
            - Machine learning model classifying high-volume financial transactions as fraudulent or legitimate.
            - Handled extreme class imbalance using SMOTE and feature scaling.
            - Exposed via high-throughput FastAPI REST API and deployed live on Render cloud.
            """
        )
        st.markdown("---")

    # 7. Machine Failure Prediction
    if proj_category in ["All Projects", "Predictive ML & Data Science"]:
        st.markdown(
            """
            ### 7. Machine Failure Predictive Maintenance System [🔗](https://github.com/ShivaKumar338/Machine-Failure-Prediction-Project)
            **Industrial IoT ML** | `Python` `Scikit-Learn` `Pandas` `Seaborn` `IoT Telemetry`
            - Predicts industrial machinery breakdowns using operational sensor data (heat dissipation, torque, vibration, load).
            - Formulates proactive maintenance scheduling, reducing simulated industrial downtime by **~15%**.
            """
        )
        st.markdown("---")

    # 8. SAKHA-AI
    if proj_category in ["All Projects", "Autonomous & Agentic AI"]:
        st.markdown(
            """
            ### 8. SAKHA-AI — Generative AI Conversational Assistant [🔗](https://github.com/ShivaKumar338/SAKHA-AI)
            **Generative AI Assistant** | `Flask` `Google Gemini API` `Python` `JavaScript`
            - Web chatbot powered by Google Gemini AI with prompt optimization, contextual dialogue handling, and responsive client.
            """
        )
        st.markdown("---")

    # 9. NeetCode Submissions
    if proj_category in ["All Projects", "Full-Stack & Systems"]:
        st.markdown(
            """
            ### 9. Algorithmic Problem Solving & LeetCode Practice [🔗](https://leetcode.com/u/shivakumar338/)
            **Computer Science Fundamentals** | `Python` `Data Structures & Algorithms`
            - Systematic solutions to 100+ NeetCode and LeetCode problems covering Dynamic Programming, Graph Traversals, Trees, and Heaps.
            - 👉 **[Explore LeetCode Profile: @shivakumar338](https://leetcode.com/u/shivakumar338/)**
            """
        )

# ==========================================
# Section 4: Skills & Tech Stack
# ==========================================
elif selection == "📊 Skills & Tech Stack":
    st.header("📊 Skills & Technical Competencies")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.subheader("🧠 AI, Deep Learning & Vision")
        st.markdown("""
        - **Python (OOP, Async, Concurrency):** Advanced (95%)
        - **PyTorch & TensorFlow / Keras:** Advanced (88%)
        - **Computer Vision (YOLOv8, OpenCV, TrOCR):** Proficient (86%)
        - **Scikit-Learn & Classical ML:** Advanced (92%)
        - **Data Analysis (Pandas, NumPy, Matplotlib, Seaborn):** Advanced (90%)
        """)

        st.subheader("⚡ Agentic AI & LLMs")
        st.markdown("""
        - **CrewAI & Multi-Agent Frameworks:** Advanced (90%)
        - **Google Gemini 1.5 Models & APIs:** Advanced (94%)
        - **RAG (Retrieval-Augmented Generation):** Proficient (85%)
        - **Prompt Engineering & Structured JSON:** Advanced (92%)
        """)

    with col_s2:
        st.subheader("⚙️ Backend & System Design")
        st.markdown("""
        - **FastAPI & Async APIs:** Advanced (92%)
        - **PostgreSQL & Supabase Architecture:** Proficient (87%)
        - **WebSockets & Real-Time Sync:** Proficient (84%)
        - **Flask & SQLite:** Advanced (90%)
        - **Celery & Redis Worker Queues:** Working Knowledge (80%)
        """)

        st.subheader("🌐 Frontend & Developer Tooling")
        st.markdown("""
        - **Modern JavaScript (ES6+), HTML5, CSS3:** Advanced (88%)
        - **React & Next.js 16 (App Router):** Proficient (82%)
        - **Git & GitHub CI/CD:** Advanced (92%)
        - **Data Structures & Algorithms (NeetCode):** Proficient (85%)
        """)

# ==========================================
# Section 5: Contact & Connect
# ==========================================
elif selection == "📫 Contact & Connect":
    st.header("📫 Get in Touch")
    st.markdown("I am actively interviewing for **Software Engineering** and **AI/ML Internship** positions.")

    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        st.markdown(
            """
            ### Direct Channels:
            - 📧 **Email:** [mshivakumar1289@gmail.com](mailto:mshivakumar1289@gmail.com)
            - 💼 **LinkedIn:** [shiva-kumar-madagoni-627038327](https://www.linkedin.com/in/shiva-kumar-madagoni-627038327/)
            - 🐙 **GitHub:** [ShivaKumar338](https://github.com/ShivaKumar338)
            - 📍 **Location:** Greater Hyderabad Area, Telangana, India
            """
        )
        st.success("✅ Fastest response via Email or LinkedIn message within 24 hours.")

    with col_c2:
        st.subheader("Send Quick Note")
        name = st.text_input("Your Name / Organization:")
        sender_email = st.text_input("Your Email:")
        role_type = st.selectbox("Role Category:", ["AI / ML Internship", "Software Engineering Internship", "Full-Stack Internship", "General Inquiry"])
        msg = st.text_area("Message / Opportunity Details:")

        if st.button("Generate Mailto Link"):
            if name and sender_email and msg:
                subject = f"[{role_type}] Internship Inquiry from {name}"
                body = f"Name: {name}\nEmail: {sender_email}\nRole: {role_type}\n\nMessage:\n{msg}"
                mailto_url = f"mailto:mshivakumar1289@gmail.com?subject={subject}&body={body}"
                st.markdown(f'<a href="{mailto_url}" target="_blank" class="status-chip" style="display:block; text-align:center;">👉 Click here to send via Email Client</a>', unsafe_allow_html=True)
            else:
                st.warning("Please fill out all fields.")
