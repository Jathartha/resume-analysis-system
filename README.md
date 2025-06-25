# Resume Analyzer System

## 🌐 Live Demo

Try the app here: [https://resume-analysis-system-v1.streamlit.app/]

A modern, AI-powered web app for analyzing resumes and matching them to job descriptions. Built with Streamlit and Cohere, this tool provides instant feedback, ATS checks, skill gap analysis, and more—all in a privacy-friendly, mobile-ready interface.

---

## 🚀 Features

- **Resume Score (0–100):** Based on relevance, keyword match, formatting, and ATS-friendliness
- **Keyword Matching:** Highlights present and missing skills/terms compared to the job description
- **ATS Compatibility Check:** Flags images, columns, tables, missing sections, and unreadable formats
- **Skill Gap Detection:** Identifies missing technical and soft skills
- **Section-Wise Feedback:** Suggestions for Experience, Education, Summary, Skills, Projects
- **Bullet Point Quality Check:** Action verbs, quantified impact, clarity, conciseness
- **Job Match Percentage:** How well your resume fits a specific job description
- **Grammar & Readability Suggestions:** Highlights complex, passive, or unclear sentences
- **Quantified Impact Suggestions:** Recommends adding metrics
- **Soft Skills Analysis & Red Flag Detection**
- **Privacy-Safe:** No resumes are stored—everything is processed in-memory
- **Beautiful, dark-themed UI with visual scoring gauge**

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit
- **AI/NLP:** Cohere API
- **Parsing:** pdfplumber, docx2txt
- **Visualization:** matplotlib
- **Other:** Python, regex, spacy

---

## ⚡ Quick Start (Local)

1. **Clone the repo:**
   ```sh
   git clone https://github.com/Jathartha/resume-analysis-system.git
   cd resume-analysis-system
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Add your Cohere API key:**
   - Create a file `.streamlit/secrets.toml` with:
     ```
     COHERE_API_KEY = "your-cohere-api-key-here"
     ```
4. **Run the app:**
   ```sh
   streamlit run app.py
   ```
5. **Open in browser:**
   - Go to `http://localhost:8501` (or the URL shown in your terminal)

---

## ☁️ Deploy on Streamlit Community Cloud

1. **Push your code to GitHub** (already done!)
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)** and sign in
3. **Create a new app:**
   - Select your repo and branch
   - Set the main file path to `app.py`
4. **Add your Cohere API key:**
   - In the app settings, go to **Secrets** and add:
     ```
     COHERE_API_KEY = "your-cohere-api-key-here"
     ```
5. **Deploy and share your app!**

---

## 📝 Usage

- Upload your resume (PDF, DOCX, or TXT)
- Paste a job description
- Click **Analyze Resume Against Job Description**
- View your match score, keyword analysis, ATS feedback, and improvement tips

---

## 🔒 Privacy

- **No resumes are stored.** All processing is done in-memory and is deleted after analysis.
- No files are saved to disk or shared with third parties.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📧 Contact

- **Author:** Jathartha
- **GitHub:** [Jathartha](https://github.com/Jathartha)
- **Project Repo:** [resume-analysis-system](https://github.com/Jathartha/resume-analysis-system)

---

Enjoy your AI-powered resume analysis!
