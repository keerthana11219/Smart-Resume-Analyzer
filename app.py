import streamlit as st
import pdfplumber
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import base64
import plotly.graph_objects as go

# -------------------- NLTK --------------------
nltk.download('stopwords')
from nltk.corpus import stopwords

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="SMART RESUME ANALYSER", layout="wide")

# -------------------- DARK MODE TOGGLE --------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_mode_checkbox")
st.session_state.dark_mode = dark_mode

st.markdown("""
<style>
div.stCheckbox { position: fixed; top: 15px; right: 20px; z-index: 999; font-size: 16px; }
body { background-color: #F4F7FA; color: #1C1C1C; }
.dark-mode body { background-color: #0E1117 !important; color: #FAFAFA !important; }
.dark-mode h1, .dark-mode h2, .dark-mode h3, .dark-mode h4, .dark-mode h5 { color: #90CAF9 !important; }
.dark-mode textarea, .dark-mode input { background-color:#1E1E1E !important; color:white !important; }
h1, h2, h3, h4, h5 { font-family: 'Arial', sans-serif; }
.logo-title { display:flex; align-items:center; justify-content:center; gap:16px; margin-top:30px; }
div.stButton > button { 
    background: linear-gradient(90deg,#1976D2,#42A5F5);
    color:white; height:50px; width:220px; border-radius:12px; font-size:16px; font-weight:bold;
    display:block; margin-left:auto; margin-right:auto;
}
</style>
""", unsafe_allow_html=True)

if dark_mode:
    st.markdown("""<script>document.querySelector('body').classList.add('dark-mode');</script>""", unsafe_allow_html=True)

# -------------------- LOGO FUNCTION --------------------
def add_logo(image_path):
    with open(image_path,"rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    return encoded

logo_base64 = add_logo("logo.png")  # Ensure your logo file is in the same folder

# -------------------- LOGO + TITLE --------------------
st.markdown(f"""
<div class="logo-title">
    <img src="data:image/png;base64,{logo_base64}" width="240">
    <div>
        <h1 style="margin:0; font-size:44px; font-weight:800;
            background:linear-gradient(90deg,#1976D2,#42A5F5,#7E57C2);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            SMART RESUME ANALYSER
        </h1>
        <p style="margin-top:4px;color:gray;">
            Analyze your resume like a real ATS and get smart suggestions
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# -------------------- DATABASES --------------------
skills_db = {
    "python": ["python"], "java": ["java"], "sql": ["sql"],
    "machine learning": ["machine learning", "ml"], "deep learning": ["deep learning", "dl"],
    "nlp": ["nlp"], "data science": ["data science"], "cloud": ["aws", "azure", "cloud"],
    "react": ["react"], "node": ["node", "nodejs"], "flask": ["flask"],
    "django": ["django"], "git": ["git"], "docker": ["docker"]
}

project_keywords = ["project", "capstone", "internship", "research"]

# -------------------- FUNCTIONS --------------------
def extract_text_from_pdf(pdf):
    text = ""
    with pdfplumber.open(pdf) as p:
        for page in p.pages:
            text += page.extract_text() or ""
    return text.lower()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    return " ".join([w for w in words if w not in stop_words])

def extract_skills(text):
    found = []
    for skill, aliases in skills_db.items():
        for a in aliases:
            if a in text:
                found.append(skill)
    return list(set(found))

def extract_projects(text):
    lines = text.split("\n")
    return [l for l in lines if any(k in l for k in project_keywords)]

def calculate_similarity(resume, jd):
    vec = TfidfVectorizer()
    vectors = vec.fit_transform([resume, jd])
    return round(cosine_similarity(vectors[0], vectors[1])[0][0] * 100, 2)

def ats_score(sim, skill, proj):
    return round(0.5*sim + 0.3*skill + 0.2*proj, 2)

def top_missing_keywords(resume, jd):
    r = set(resume.split())
    j = jd.split()
    miss = [w for w in j if w not in r]
    return [k for k, v in Counter(miss).most_common(5)]

def suggest_roles(skills):
    roles = []
    if "machine learning" in skills or "data science" in skills:
        roles += ["Data Scientist", "ML Engineer"]
    if "nlp" in skills:
        roles.append("NLP Engineer")
    if "cloud" in skills:
        roles.append("Cloud Engineer")
    if not roles:
        roles.append("Software Engineer")
    return list(set(roles))

# -------------------- MAIN LAYOUT --------------------
left_col, right_col = st.columns([1.2, 2])

# LEFT COLUMN
with left_col:
    st.subheader("📌 About Project")
    st.markdown("""
Projects are a practical way to showcase your skills and experience.
They demonstrate how you apply knowledge to real-world problems.
""")
    st.markdown("**Importance:**")
    st.markdown("""
- ✅ Show hands-on experience with relevant technologies.
- ✅ Highlight problem-solving and critical thinking skills.
- ✅ Make your resume stand out from purely academic profiles.
- ✅ Demonstrate ability to work on real-world or complex datasets.
- ✅ Can showcase specialized areas like Machine Learning, NLP, or Cloud.
""")
    st.markdown('<div style="height:550px;"></div>', unsafe_allow_html=True)

# RIGHT COLUMN
with right_col:
    st.subheader("📄 Upload Resume (PDF)")
    resume_file = st.file_uploader("", type=["pdf"], key="resume_uploader")

    st.subheader("📝 Job Description")
    job_desc = st.text_area("", height=250, key="job_desc_textarea")

    st.markdown("<div style='margin-top:-10px;'></div>", unsafe_allow_html=True)

    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False

    if st.button("Analyze Resume", key="analyze_resume_btn"):
        st.session_state.analyze_clicked = True

# -------------------- ANALYSIS --------------------
if st.session_state.analyze_clicked:
    if resume_file and job_desc:
        resume_text = clean_text(extract_text_from_pdf(resume_file))
        jd_text = clean_text(job_desc)

        similarity = calculate_similarity(resume_text, jd_text)
        r_skills = extract_skills(resume_text)
        j_skills = extract_skills(jd_text)

        matched_skills = list(set(r_skills) & set(j_skills))
        missing_skills = list(set(j_skills) - set(r_skills))

        r_proj = extract_projects(resume_text)
        j_proj = extract_projects(jd_text)

        matched_proj = list(set(r_proj) & set(j_proj))
        missing_proj = list(set(j_proj) - set(r_proj))

        skill_score = round(len(matched_skills)/max(len(j_skills),1)*100,2)
        proj_score = round(len(matched_proj)/max(len(j_proj),1)*100,2)
        final_ats = ats_score(similarity, skill_score, proj_score)

        # -------------------- DASHBOARD --------------------
        st.subheader("📊 ATS Dashboard")
        col1_dash, col2_dash, col3_dash, col4_dash = st.columns(4)
        card_style = "padding:15px;border-radius:12px;background:#1976D2;color:white;text-align:center;"
        col1_dash.markdown(f"<div style='{card_style}'><h4>Similarity</h4><h2>{similarity}%</h2></div>", unsafe_allow_html=True)
        col2_dash.markdown(f"<div style='{card_style}'><h4>Skill Match</h4><h2>{skill_score}%</h2></div>", unsafe_allow_html=True)
        col3_dash.markdown(f"<div style='{card_style}'><h4>Project Match</h4><h2>{proj_score}%</h2></div>", unsafe_allow_html=True)
        col4_dash.markdown(f"<div style='{card_style}'><h4>Final ATS</h4><h2>{final_ats}%</h2></div>", unsafe_allow_html=True)

        # -------------------- GAUGE --------------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_ats,
            title={'text': "ATS Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1976D2"},
                'steps': [
                    {'range': [0, 50], 'color': "#FFCDD2"},
                    {'range': [50, 75], 'color': "#FFE082"},
                    {'range': [75, 100], 'color': "#C8E6C9"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': final_ats}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # -------------------- SUGGESTIONS --------------------
        st.subheader("💡 Skills & Projects Suggestions")
        c1_sug, c2_sug = st.columns(2)
        with c1_sug:
            st.markdown("**Matched Skills:**")
            if matched_skills:
                for skill in matched_skills:
                    st.markdown(f"<span style='color:green;font-weight:bold'>{skill}</span>", unsafe_allow_html=True)
            else:
                st.write("None")
            st.markdown("**Missing Skills:**")
            if missing_skills:
                for skill in missing_skills:
                    st.markdown(f"<span style='color:red;font-weight:bold'>{skill}</span>", unsafe_allow_html=True)
                    st.markdown(f"- Suggest: Work on projects using **{skill}**")
            else:
                st.write("None")
        with c2_sug:
            st.markdown("**Matched Projects:**")
            if matched_proj:
                for proj in matched_proj:
                    st.markdown(f"<span style='color:green;font-weight:bold'>{proj}</span>", unsafe_allow_html=True)
            else:
                st.write("None")
            st.markdown("**Missing Projects:**")
            if missing_proj:
                for proj in missing_proj:
                    st.markdown(f"<span style='color:red;font-weight:bold'>{proj}</span>", unsafe_allow_html=True)
            else:
                st.write("None")

        # -------------------- TOP MISSING KEYWORDS --------------------
        st.subheader("🏷 Top Missing JD Keywords")
        top_keywords = top_missing_keywords(resume_text, jd_text)
        if top_keywords:
            st.markdown(", ".join([f"<span style='color:red'>{k}</span>" for k in top_keywords]), unsafe_allow_html=True)
        else:
            st.write("None")

        # -------------------- SUGGESTED JOB ROLES --------------------
        st.subheader("💼 Suggested Job Roles")
        st.write(", ".join(suggest_roles(r_skills)))

    else:
        st.error("Please upload resume and paste job description")
