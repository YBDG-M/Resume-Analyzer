# Resume Analyzer

**Automated Resume–Job Matching using NLP & HR Analytics**

Resume Analyzer ini dibangun untuk membantu HR dan job seekers melakukan penilaian objektif terhadap kecocokan CV dengan job description secara otomatis. Sistem menggunakan NLP, semantic similarity, dan evidence-based HR metrics untuk menghasilkan skor kecocokan serta rekomendasi peningkatan CV.

---

## Deskripsi Singkat

* Preprocessed & extracted text from resumes & job descriptions (PDF/DOCX)
* Analyzed **hard skills, soft skills, experience**, dan **ATS keywords** menggunakan NLP
* Menghitung **resume–job fit score** berdasarkan keyword overlap, semantic similarity, dan weighted scoring
* Visualized insights & generated **CV improvement recommendations** via Streamlit
* Exported **PDF reports** untuk kandidat maupun perusahaan

---

## Project Links

**GitHub Repository**
[https://github.com/YBDG-M/Resume-Analyzer](https://github.com/YBDG-M/Resume-Analyzer)

**Notion Workspace**
[https://grand-sink-25e.notion.site/Resume-Analyzer-2bc817c0e2c98010a4c6f275c95dd543](https://grand-sink-25e.notion.site/Resume-Analyzer-2bc817c0e2c98010a4c6f275c95dd543)

**Google Slides Deck**
[https://docs.google.com/presentation/d/1Jiu1S5pdOXdHQ-b1VJSOWBlT6c7-EvhQuxCWtgEyzF8/edit?usp=sharing](https://docs.google.com/presentation/d/1Jiu1S5pdOXdHQ-b1VJSOWBlT6c7-EvhQuxCWtgEyzF8/edit?usp=sharing)

**Streamlit App (Live Demo)**
[https://resume-analyzer-yogibimadwigraham.streamlit.app/](https://resume-analyzer-yogibimadwigraham.streamlit.app/)

---

## Key Features

### 1. Resume & Job Description Parsing

* Extract text from **PDF & DOCX** files
* Automatic cleaning: lowercase, punctuation removal, stopwords, lemmatization

### 2. NLP-Based Skill & Experience Matching

* Hard skill extraction (regex + embeddings)
* Soft skill extraction (dictionary + semantic similarity)
* Experience detection (NER untuk organisasi, role, duration)

### 3. Resume–Job Fit Score

Menggunakan tiga komponen utama:

1. **Keyword Overlap Score**
2. **Semantic Similarity Score (sentence-transformers)**
3. **Weighted Job Match Score**

### 4. Insights & Visualizations (Streamlit)

* Skill match visualization
* Missing keywords
* Experience fit timeline
* Soft-skill radar chart

### 5. Auto-Generated PDF Report

* Summary
* Insights per kategori
* Personalized recommendations

---

## Architecture / Workflow

```
Upload Resume (PDF/DOCX)
            ↓
Text Extraction → Cleaning → NLP Processing (spaCy + Transformers)
            ↓
Skill Extraction & Matching
            ↓
Semantic Similarity (SBERT)
            ↓
Weighted Scoring (Hard Skill, Experience, ATS)
            ↓
Visualization + Recommendations
            ↓
Export PDF Report
```

---

## Tech Stack

**Languages / Libraries**

* Python (Pandas, NumPy)
* spaCy
* sentence-transformers
* sklearn
* matplotlib / seaborn
* PyMuPDF / python-docx

**Application**

* Streamlit
* PDF Report Generation (ReportLab / FPDF)

**HR Analytics**

* competency mapping
* ATS keyword modeling
* matching algorithms

## Output (Report)

* Skill Match Score
* Experience Fit (role & duration)
* Missing ATS keywords
* Soft Skill Evidence
* Final Resume–Job Fit Score
* Recommendations

---

## Why This Project Matters?

* Membantu HR mempercepat screening ratusan CV
* Memberikan objektivitas berdasarkan HR science
* Mendukung jobseekers memperbaiki resume secara terarah
* Menggunakan pendekatan **evidence-based** (literature: Schmidt & Hunter, 1998; LinkedIn Talent Insights, 2022)

---

## References

* Schmidt, F. L., & Hunter, J. E. (1998). *The validity and utility of selection methods in personnel psychology.*
* SHRM Competency Model
* ESCO Skill Taxonomy
* LinkedIn Global Talent Report

---

## Author

**Yogi Bima Dwi Graha M.**
NLP · HR Analytics · Data Engineering
GitHub: [https://github.com/YBDG-M](https://github.com/YBDG-M)
