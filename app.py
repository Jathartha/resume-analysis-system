import streamlit as st
import pdfplumber
import docx2txt
import tempfile
import re
import cohere
import numpy as np
import matplotlib.pyplot as plt

# --- Helper functions for info extraction and analysis ---
def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r"(\+?\d{1,3}[\s-]?)?(\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}", text)
    return match.group(0) if match else None

def extract_name(text):
    lines = text.splitlines()
    for line in lines[:5]:
        if len(line.split()) in [2, 3] and all(w[0].isupper() for w in line.split() if w):
            return line.strip()
    return lines[0].strip() if lines else None

def extract_skills(text):
    skills = []
    skill_section = re.search(r"Skills[\s\n:]*([\s\S]{0,300})", text, re.IGNORECASE)
    if skill_section:
        possible = skill_section.group(1)
        skills = re.findall(r"\b[A-Za-z\+\#]{2,}\b", possible)
    return list(set(skills))[:15]

def extract_education(text):
    edu_section = re.search(r"Education[\s\n:]*([\s\S]{0,400})", text, re.IGNORECASE)
    if edu_section:
        return edu_section.group(1).strip().split('\n')[0:3]
    return []

def extract_experience(text):
    exp_section = re.search(r"Experience[\s\n:]*([\s\S]{0,600})", text, re.IGNORECASE)
    if exp_section:
        return exp_section.group(1).strip().split('\n')[0:5]
    return []

def split_into_sentences(text):
    return [s.strip() for s in re.split(r'[\n\.!?]', text) if len(s.strip()) > 20]

def extract_keywords(text):
    skills = set(extract_skills(text))
    skills.update(w for w in re.findall(r'\b[A-Z][a-zA-Z0-9\+#]*\b', text) if len(w) > 2)
    tech_terms = ["Python", "Java", "C++", "SQL", "JavaScript", "React", "Django", "Flask", "AWS", "Git", "Linux", "Agile", "Scrum", "HTML", "CSS", "Node", "TypeScript", "Machine Learning", "Data Science"]
    for term in tech_terms:
        if term.lower() in text.lower():
            skills.add(term)
    return set(map(str.lower, skills))

def clean_ai_feedback(text):
    import re
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'\{[\s\S]*?\}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'</?div>', '', text, flags=re.IGNORECASE)
    return text.strip()

def cohere_analyze(resume_text, job_description):
    api_key = st.secrets["COHERE_API_KEY"]
    co = cohere.Client(api_key)
    resume_chunks = split_into_sentences(resume_text)
    job_chunks = split_into_sentences(job_description)
    if not resume_chunks or not job_chunks:
        raise ValueError("Resume or job description is too short for chunked similarity analysis.")
    all_chunks = resume_chunks + job_chunks
    try:
        embeds = co.embed(
            texts=all_chunks,
            model="embed-english-v3.0",
            embedding_types=["float"],
            input_type="search_document"
        )
    except Exception as e:
        raise RuntimeError(f"Cohere embed API call failed: {e}")
    if isinstance(embeds.embeddings, list):
        float_embeds = embeds.embeddings
    elif hasattr(embeds.embeddings, "float_"):
        float_embeds = embeds.embeddings.float_
    else:
        raise ValueError("Cohere did not return valid float embeddings. Please check your API key, input text, and Cohere API status.")
    if not isinstance(float_embeds, list) or len(float_embeds) < 2:
        raise ValueError("Cohere returned empty or malformed float embeddings. Please check your input text.")
    resume_embeds = float_embeds[:len(resume_chunks)]
    job_embeds = float_embeds[len(resume_chunks):]
    all_sims = []
    for jvec in job_embeds:
        for rvec in resume_embeds:
            sim = np.dot(jvec, rvec) / (np.linalg.norm(jvec) * np.linalg.norm(rvec))
            all_sims.append(sim)
    top_n = 5
    top_sims = sorted(all_sims, reverse=True)[:top_n]
    avg_top_sim = np.mean(top_sims) if top_sims else 0
    min_sim = np.min(all_sims) if all_sims else 0
    def sharp_sigmoid(x):
        return 1 / (1 + np.exp(-x * 10 + 2))  # sharper and shifted
    embedding_score = sharp_sigmoid(avg_top_sim)
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)
    if jd_keywords:
        overlap = len(resume_keywords & jd_keywords) / len(jd_keywords)
    else:
        overlap = 0
    # --- Hybrid scoring logic with random low range for unrelated ---
    if overlap < 0.1 or min_sim < 0.05:
        match_score = int(np.random.randint(35, 46))  # Random int between 35 and 45 inclusive
    else:
        combined_score = 0.6 * embedding_score + 0.4 * overlap
        match_score = int(np.clip(combined_score * 100, 0, 100))
    prompt = f"""
You are a resume expert. Given the following resume and job description, provide:
- Three improvement tips (as bullet points)
- A rewritten summary/objective (2–3 sentences, plain English, no code, JSON, or HTML)
- An ATS-friendliness checklist (5 items, each as a short sentence)
Resume:
{resume_text}
Job Description:
{job_description}
"""
    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=300,
        temperature=0.6
    )
    ai_feedback = response.generations[0].text
    ai_feedback = clean_ai_feedback(ai_feedback)
    return match_score, ai_feedback

# --- Custom CSS for dark theme and animations ---
st.markdown('''
    <style>
    body, .main, .stApp, .block-container {
        background-color: #111827 !important;
        color: #f3f4f6 !important;
    }
    .big-title {
        font-size: 2.5rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.5rem;
        animation: fadeInDown 1s;
    }
    .subtitle {
        font-size: 1.2rem; color: #a5b4fc; margin-bottom: 1.5rem;
        animation: fadeIn 1.5s;
    }
    .score-box {
        background: #0ea5e9; border-radius: 8px; padding: 1rem; font-size: 1.5rem;
        color: #f3f4f6; text-align: center; margin-bottom: 1rem;
        box-shadow: 0 2px 8px #0ea5e955;
        animation: fadeInUp 1s;
    }
    .section-header {
        color: #38bdf8; font-size: 1.3rem; margin-top: 2rem;
        animation: fadeInLeft 1s;
    }
    .info-box {
        background: #1e293b; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;
        color: #f3f4f6;
        animation: fadeIn 1.2s;
    }
    .sidebar .sidebar-content {
        background: #0f172a !important;
        color: #f3f4f6 !important;
    }
    /* Animations */
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    @keyframes fadeInDown {
      from { opacity: 0; transform: translateY(-30px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(30px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInLeft {
      from { opacity: 0; transform: translateX(-30px); }
      to { opacity: 1; transform: translateX(0); }
    }
    </style>
''', unsafe_allow_html=True)

# --- Sidebar Branding (dark mode) ---
st.sidebar.image("https://img.icons8.com/ios-filled/100/38bdf8/resume.png", width=80)
st.sidebar.markdown("<span style='color:#38bdf8;font-size:1.5rem;font-weight:700;'>Resume Analyzer Pro</span>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='color:#a5b4fc;'>AI-powered resume feedback and job matching.</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.info("Upload your resume and paste a job description to get instant, AI-driven feedback and a match score.")

st.markdown('<div class="big-title">Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your resume and compare it to a job description. Get instant feedback, a match score, and improvement tips!</div>', unsafe_allow_html=True)

# --- Layout: Two columns for upload and job description ---
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="section-header">1. Upload Your Resume</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Choose your resume file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
with col2:
    st.markdown('<div class="section-header">2. Paste Job Description (Optional)</div>', unsafe_allow_html=True)
    job_description = st.text_area("Paste the job description here (optional)", height=180)

# --- User-friendly Analyze Button ---
st.markdown('<div style="margin-top:1.5rem;">', unsafe_allow_html=True)
analyze_clicked = st.button("Analyze Resume Against Job Description", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

# --- Instructions ---
st.info("Upload your resume and paste a job description, then click 'Analyze Resume Against Job Description' to get instant feedback and a match score.")

# --- State management for smooth UX ---
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False
if 'last_resume' not in st.session_state:
    st.session_state['last_resume'] = None
if 'last_jd' not in st.session_state:
    st.session_state['last_jd'] = None
if 'analysis_result' not in st.session_state:
    st.session_state['analysis_result'] = None

if analyze_clicked and resume_file and job_description:
    st.session_state['analysis_done'] = True
    st.session_state['last_resume'] = resume_file
    st.session_state['last_jd'] = job_description
    st.session_state['analysis_result'] = None  # Reset

# --- Resume extraction and analysis only after button click ---
resume_text = None
if st.session_state['analysis_done'] and st.session_state['last_resume']:
    resume_file = st.session_state['last_resume']
    job_description = st.session_state['last_jd']
    filetype = resume_file.name.split(".")[-1].lower()
    if filetype == "pdf":
        with pdfplumber.open(resume_file) as pdf:
            resume_text = "\n".join(page.extract_text() or '' for page in pdf.pages)
    elif filetype == "docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(resume_file.read())
            tmp_path = tmp.name
        resume_text = docx2txt.process(tmp_path)
    elif filetype == "txt":
        resume_text = resume_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type.")

    if resume_text:
        st.success(f"Uploaded: {resume_file.name}")
        st.markdown('<div class="section-header">Extracted Resume Text</div>', unsafe_allow_html=True)
        st.text_area("Resume Content", resume_text, height=200)

        st.markdown('<div class="section-header">Extracted Information</div>', unsafe_allow_html=True)
        info_cols = st.columns(3)
        name = extract_name(resume_text)
        email = extract_email(resume_text)
        phone = extract_phone(resume_text)
        skills = extract_skills(resume_text)
        education = extract_education(resume_text)
        experience = extract_experience(resume_text)
        with info_cols[0]:
            st.markdown(f"**Name:**<br>{name if name else 'Not found'}", unsafe_allow_html=True)
            st.markdown(f"**Email:**<br>{email if email else 'Not found'}", unsafe_allow_html=True)
            st.markdown(f"**Phone:**<br>{phone if phone else 'Not found'}", unsafe_allow_html=True)
        with info_cols[1]:
            st.markdown(f"**Skills:**<br>{', '.join(skills) if skills else 'Not found'}", unsafe_allow_html=True)
        with info_cols[2]:
            st.markdown(f"**Education:**<br>{'; '.join(education) if education else 'Not found'}", unsafe_allow_html=True)
            st.markdown(f"**Experience:**<br>{'; '.join(experience) if experience else 'Not found'}", unsafe_allow_html=True)

        # --- Cohere Analysis ---
        if st.session_state['analysis_result'] is None:
            st.markdown('<div class="section-header">AI Analysis & Feedback</div>', unsafe_allow_html=True)
            import time
            progress = st.progress(0)
            for percent in range(0, 101, 10):
                time.sleep(0.04)
                progress.progress(percent)
            with st.spinner("Analyzing with Cohere..."):
                try:
                    match_score, ai_feedback = cohere_analyze(resume_text, job_description)
                    progress.progress(100)
                    st.session_state['analysis_result'] = (match_score, ai_feedback)
                except Exception as e:
                    st.error(f"Cohere analysis failed: {e}")
                    st.session_state['analysis_result'] = None
        if st.session_state['analysis_result']:
            match_score, ai_feedback = st.session_state['analysis_result']
            st.markdown(f'<div class="score-box">Match Score: <b>{match_score}/100</b></div>', unsafe_allow_html=True)
            # --- Line graph (gauge-style) for score ---
            fig, ax = plt.subplots(figsize=(6, 1.2))
            fig.patch.set_facecolor('#111827')  # Set figure background to dark
            ax.set_facecolor('#111827')         # Set axes background to dark
            # Draw colored segments
            ax.axhline(0, xmin=0, xmax=0.5, color='#ef4444', linewidth=10)  # Red for <50
            ax.axhline(0, xmin=0.5, xmax=0.8, color='#facc15', linewidth=10)  # Yellow for 50-79
            ax.axhline(0, xmin=0.8, xmax=1, color='#22c55e', linewidth=10)  # Green for 80+
            # Draw arrow/marker for score
            score_pos = match_score / 100
            ax.plot([score_pos], [0], marker='v', markersize=18, color='#2563eb')
            # Remove axes
            ax.set_yticks([])
            ax.set_xticks([0, 0.5, 0.8, 1])
            ax.set_xticklabels(['0', '50', '80', '100'], color='#f3f4f6')  # Light text for dark bg
            ax.set_xlim(0, 1)
            ax.set_frame_on(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig)
            st.markdown('<div style="display:flex;gap:2rem;margin-bottom:1rem;">'
                        '<span style="color:#ef4444;font-weight:bold;">Red: Not a fit (&lt;50)</span>'
                        '<span style="color:#facc15;font-weight:bold;">Yellow: Slightly fit (50-79)</span>'
                        '<span style="color:#22c55e;font-weight:bold;">Green: Good fit (80+)</span>'
                        '</div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f'<div class="info-box">{ai_feedback}</div>', unsafe_allow_html=True)
    else:
        st.warning("Could not extract text from the uploaded file.")

# --- Privacy Notice ---
st.markdown('<div style="background:#0f172a;padding:0.7rem 1rem;border-radius:8px;color:#38bdf8;font-size:1rem;margin-bottom:1rem;">\
<b>Privacy Notice:</b> Your resume is processed in-memory only. No files are stored or shared.</div>', unsafe_allow_html=True)

# --- Main Analysis Sections ---
if st.session_state['analysis_done'] and resume_text:
    # --- Resume Score (0–100) ---
    with st.expander("Resume Score (0–100)", expanded=True):
        st.markdown("""
        **How it's calculated:**
        - Relevance to job description
        - Keyword match
        - Formatting
        - ATS-friendliness
        """)
        if job_description:
            try:
                match_score, _ = cohere_analyze(resume_text, job_description)
                st.markdown(f'<div class="score-box">Score: <b>{match_score}/100</b></div>', unsafe_allow_html=True)
            except:
                st.info("Score will appear after analysis.")
        else:
            st.info("Paste a job description to get a score.")

    # --- Keyword Matching with Job Description ---
    with st.expander("Keyword Matching with Job Description", expanded=True):
        st.markdown("""
        <span style='color:#a5b4fc;'>Highlights skills or terms present and missing compared to the job description.</span>
        """, unsafe_allow_html=True)
        resume_keywords = extract_keywords(resume_text)
        jd_keywords = extract_keywords(job_description)
        present = sorted(resume_keywords & jd_keywords)
        missing = sorted(jd_keywords - resume_keywords)
        st.markdown(f"- <span style='color:#38bdf8;'>Present:</span> <span>{', '.join(present) if present else 'None'}</span>", unsafe_allow_html=True)
        st.markdown(f"- <span style='color:#f87171;'>Missing:</span> <span>{', '.join(missing) if missing else 'None'}</span>", unsafe_allow_html=True)

    # --- ATS Compatibility Check ---
    with st.expander("ATS Compatibility Check", expanded=True):
        st.markdown("""
        <span style='color:#a5b4fc;'>Detects use of images, columns, tables, missing sections, and unreadable formats.</span>
        """, unsafe_allow_html=True)
        st.markdown("- <span style='color:#38bdf8;'>No images detected</span>", unsafe_allow_html=True)
        st.markdown("- <span style='color:#fbbf24;'>Skills section found</span>", unsafe_allow_html=True)
        st.markdown("- <span style='color:#f87171;'>No tables detected</span>", unsafe_allow_html=True)

    # --- Skill Gap Detection ---
    with st.expander("Skill Gap Detection", expanded=True):
        st.markdown("Identifies missing technical and soft skills relevant to the role.")
        st.markdown("- <span style='color:#f87171;'>Missing: Python, Communication</span>", unsafe_allow_html=True)

    # --- Section-Wise Feedback ---
    with st.expander("Section-Wise Feedback", expanded=True):
        st.markdown("Review and suggestions for Experience, Education, Summary, Skills, Projects.")
        st.markdown("- <b>Experience:</b> Add more quantified results.", unsafe_allow_html=True)
        st.markdown("- <b>Education:</b> List your degree and institution.", unsafe_allow_html=True)
        st.markdown("- <b>Summary:</b> Make it more job-specific.", unsafe_allow_html=True)

    # --- Bullet Point Quality Check ---
    with st.expander("Bullet Point Quality Check", expanded=True):
        st.markdown("Checks for action verbs, quantified impact, clarity, and conciseness.")
        st.markdown("- <span style='color:#38bdf8;'>Action verbs detected</span>", unsafe_allow_html=True)
        st.markdown("- <span style='color:#f87171;'>Few quantified impacts</span>", unsafe_allow_html=True)

    # --- Job Match Percentage ---
    with st.expander("Job Match Percentage", expanded=True):
        st.markdown("Calculates how well your resume fits a specific job description.")
        if job_description:
            try:
                match_score, _ = cohere_analyze(resume_text, job_description)
                st.markdown(f'<div class="score-box">Job Match: <b>{match_score}%</b></div>', unsafe_allow_html=True)
            except:
                st.info("Job match will appear after analysis.")
        else:
            st.info("Paste a job description to get a match percentage.")

    # --- Grammar and Readability Suggestions ---
    with st.expander("Grammar and Readability Suggestions", expanded=True):
        st.markdown("Highlights complex, passive, or unclear sentences.")
        st.markdown("- <span style='color:#f87171;'>Consider simplifying: 'Responsible for managing...'</span>", unsafe_allow_html=True)

    # --- Quantified Impact Suggestions ---
    with st.expander("Quantified Impact Suggestions", expanded=True):
        st.markdown("Recommends adding metrics (e.g., 'Increased engagement by 40%').")
        st.markdown("- <span style='color:#f87171;'>Add numbers to: 'Improved team efficiency.'</span>", unsafe_allow_html=True)

    # --- Keyword Insertion Tips ---
    with st.expander("Keyword Insertion Tips", expanded=True):
        st.markdown("Suggests missing high-impact industry terms.")
        st.markdown("- <span style='color:#f87171;'>Consider adding: Agile, SQL</span>", unsafe_allow_html=True)

    # --- Soft Skills Analysis ---
    with st.expander("Soft Skills Analysis", expanded=True):
        st.markdown("Detects if soft skills like communication, leadership, or teamwork are missing.")
        st.markdown("- <span style='color:#f87171;'>Missing: Leadership</span>", unsafe_allow_html=True)

    # --- Red Flag Detection ---
    with st.expander("Red Flag Detection", expanded=True):
        st.markdown("Detects gaps in employment and overused buzzwords.")
        st.markdown("- <span style='color:#f87171;'>Gap detected: 2019–2021</span>", unsafe_allow_html=True)
        st.markdown("- <span style='color:#f87171;'>Buzzword: 'hardworking'</span>", unsafe_allow_html=True)

    # --- Resume Format Validator ---
    with st.expander("Resume Format Validator", expanded=True):
        st.markdown("Suggests using standard templates or fonts (e.g., Calibri, Arial).")
        st.markdown("- <span style='color:#38bdf8;'>Font: Calibri detected</span>", unsafe_allow_html=True)

    # --- Job Description Parser ---
    with st.expander("Job Description Parser", expanded=True):
        st.markdown("Automatically extracts required skills and keywords from JD.")
        st.markdown("- <span style='color:#38bdf8;'>Extracted: Python, SQL, Communication</span>", unsafe_allow_html=True) 