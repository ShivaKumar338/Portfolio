import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64

st.set_page_config(page_title="Shiva Kumar's Portfolio", page_icon=":tada:", layout="wide")

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Projects", "Skills", "Contact"])



skills = {
    "Python": 5,
    "Machine Learning": 5,
    "Data Analysis": 4,
    "JavaScript": 3,
    "PostgreSQL": 4,
    "TensorFlow": 4,
    "Pytorch": 4,   
    
}
if selection == "Home":
    # Read image and convert to base64
    with open("Shiva.png", "rb") as img_file:
        img_bytes = img_file.read()
        encoded = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <div style="display:flex; justify-content:center;">
            <img src="data:image/png;base64,{encoded}"
                style="
                    width:180px;
                    height:180px;
                    border-radius:50%;
                    object-fit:cover;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
                ">
        </div>
        <p style="text-align:center; font-size:18px;"><b>Shiva Kumar</b></p>
        """,
        unsafe_allow_html=True
    )
    st.title("Shiva Kumar's Portfolio")
    st.markdown("**AI Engineer | Data Scientist | Web Developer**")
    st.markdown(
        "[📧 Email](https://mail.google.com/mail/?view=cm&fs=1&to=mshivakumar1289@gmail.com) | "
        "[🔗 LinkedIn](https://www.linkedin.com/in/shiva-kumar-madagoni-627038327/) | "
        "[🐙 GitHub](https://github.com/ShivaKumar338)"
    )
    st.header("About Me")
    st.markdown(
        """
        Hello! I'm Shiva Kumar, an AI Engineer/ Ml Engineer and Data Scientist with a passion for leveraging technology to solve real-world problems.
        With expertise in Python, machine learning and data analysis, I enjoy creating innovative solutions that drive efficiency and insights.
        - I have worked on various projects ranging from AI-powered chatbots to predictive analytics systems.
        - In my free time, I love exploring new technologies, contributing to open-source projects and enhancing my skills in AI and data science.
        I'm excited to connect with like-minded professionals and explore new opportunities in the tech industry.

        """
    )
    st.subheader("Experience")
    st.markdown(
        """
        - **Training ML Intern in EdiGlobe** (October(2025) - December(2025))
            - Collaborated on machine learning projects to enhance data-driven decision-making.
            - Utilized various ML algorithms to analyze and interpret complex datasets.
            - Built machine learning models to improve business processes.
            - Learned Neural Networks, DeepLearning concepts, Decision Trees, Random Forests and other ML techniques.

        - **Training AI Intern in SkillDuniya** (March(2025) - May(2025))
            - Worked on training AI models using ScikitLearn,Pytorch,Tensorflow.
            - Developed AI models for predictive analytics and automation.
            - Leaned about advanced AI techniques and their applications.
            - This internship provided hands-on experience in AI model development and deployment.
            
        """
    )

elif selection == "Projects":
    st.header("Projects")
    st.markdown("A selection of projects demonstrating my skills in Machine Learning, AI and Data Science.")


    st.subheader("Skills Demonstrated Across Projects")
    st.markdown("""
    - Machine Learning model development and evaluation  
    - Computer Vision and Object Detection  
    - Deep Learning using TensorFlow and PyTorch  
    - Natural Language Processing and LLM integration  
    - Data analysis and predictive modeling  
    - Research-oriented problem solving and experimentation  
    """)


    # -------------------------------
    st.markdown(
    "## Vision AI Project: Object Detection System [🔗](https://github.com/ShivaKumar338/Vision-Ai-Assistance)"
    )

    st.markdown("""
    Developed a **Vision AI–based object detection system** capable of identifying and localizing objects in real time.

    - Designed and implemented object detection pipelines using **TensorFlow and OpenCV**.
    - Performed image preprocessing, model inference and post-processing for bounding box generation.
    - Optimized the system for **real-time inference on edge devices**, focusing on latency and efficiency.
    - Applied the solution to practical use cases such as **security monitoring** and **inventory management**.
    - Evaluated performance using detection accuracy and inference speed.
    """)

    # -------------------------------
    st.markdown("## Featured Project: AI-Powered Chatbot [🔗](https://github.com/ShivaKumar338/SAKHA-AI)")
    st.markdown("""
    Built an **AI-powered conversational chatbot** to assist users with common queries and tasks.
    - Implemented backend logic using **Python** and integrated **Gemini API** for natural language understanding.
    - Designed conversational flows to provide accurate and context-aware responses.
    - Integrated the chatbot with a **web interface** for a smooth and interactive user experience.
    - Focused on prompt design, response handling and user interaction optimization.
    """)

    # -------------------------------
    st.markdown("## Data Science Project: Machine Failure Prediction [🔗](https://github.com/ShivaKumar338/Machine-Failure-Prediction-Project)")
    st.markdown("""
    Developed a **predictive maintenance model** to forecast machine failures in a manufacturing environment.

    - Analyzed historical **sensor and operational data** to identify failure patterns.
    - Built and evaluated machine learning models for failure prediction.
    - Improved maintenance scheduling and decision-making, reducing **machine downtime by approximately 15%**.
    - Assessed model performance using appropriate evaluation metrics to ensure reliability.
    """)

elif selection == "Skills":
    st.markdown("""
    I possess strong skills in **Python, Machine Learning, Data Analysis, PostgreSQL, TensorFlow, PyTorch and JavaScript**.  
    I am capable of designing, training and evaluating machine learning models and building end-to-end AI solutions.

    My learning approach is **research-oriented and theory-backed**. I regularly study machine learning concepts from industry-standard resources such as  
    **_Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow_**, which strengthens my understanding of model internals, best practices and real-world applications.

    I am currently deepening my knowledge of **Deep Learning** with hands-on experimentation using **TensorFlow and PyTorch**.
    """)

    st.header("Skills Overview")
    for skill, level in skills.items():
        st.markdown(f"**{skill}**")
        st.progress(level * 20)
    df_skills = pd.DataFrame({
        'Skill': list(skills.keys()),
        'Proficiency': list(skills.values())
    })
    st.bar_chart(df_skills.set_index('Skill'))

elif selection == "Contact":
    st.header("Contact Me")
    st.markdown(
        """**Feel free to reach out to me through the following channels:**
        """
        "[📧 Email](https://mail.google.com/mail/?view=cm&fs=1&to=mshivakumar1289@gmail.com) | "
        "[🔗 LinkedIn](https://www.linkedin.com/in/shiva-kumar-madagoni-627038327/) | "
        "[🐙 GitHub](https://github.com/ShivaKumar338)"

    )
