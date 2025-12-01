# app.py
"""
Domain-Adaptive Resume Analyzer (Streamlit)
Features:
- Extract text from PDF/DOCX
- Auto-detect job category (ESCO-like simple taxonomy)
- Compute indicators: Hard Skill, Experience Fit, ATS Keyword Match, Soft Skill Relevance, Education Fit
- Explanations: what found, what missing, why score (with literature refs)
- Recommendation snippets to improve resume
- PDF export
- Uses fastembed if available; else TF-IDF fallback
"""

import streamlit as st
import re
import os
import numpy as np
from io import BytesIO
from datetime import datetime

# File parsing
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
import docx

# Optional fastembed
USE_FASTEMBED = False
try:
    from fastembed import TextEmbedding
    USE_FASTEMBED = True
except Exception:
    USE_FASTEMBED = False

# Fallback TF-IDF
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    USE_TFIDF = True
except Exception:
    USE_TFIDF = False

# PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# -------------------------
# Utilities: text extraction
# -------------------------
def extract_text_from_pdf(file_obj):
    if PyPDF2 is None:
        return ""
    try:
        reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for page in reader.pages:
            pg = page.extract_text()
            if pg:
                text += pg + " "
        return text.strip()
    except Exception:
        return ""

def extract_text_from_docx(file_obj):
    try:
        doc = docx.Document(file_obj)
        return " ".join([p.text for p in doc.paragraphs]).strip()
    except Exception:
        return ""

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        try:
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            return ""

# -------------------------
# Simple ESCO-like taxonomy
# (small curated skill lists per domain)
# -------------------------
ESCO_TAXONOMY = {
    "data_science": {
        "label": "Data / Data Science",
        "hard_skills": ["python","r","sql","pandas","numpy","scikit-learn","machine learning","deep learning","statistics","data analysis","etl","big data","spark"],
        "soft_skills": ["communication","problem solving","critical thinking","collaboration","curiosity"],
        "edu_levels": ["bachelor","s1","master","s2","phd"]
    },
    "software_engineer": {
        "label": "Software Engineering",
        "hard_skills": ["python","java","c++","git","docker","kubernetes","rest api","microservices","algorithms","data structures","ci/cd"],
        "soft_skills": ["teamwork","communication","problem solving","ownership"],
        "edu_levels": ["bachelor","s1","master"]
    },
    "product_marketing": {
        "label": "Product / Marketing",
        "hard_skills": ["marketing","seo","sem","google analytics","campaign","content","social media","brand"],
        "soft_skills": ["communication","creative thinking","stakeholder management"],
        "edu_levels": ["bachelor","s1","master"]
    },
    "finance": {
        "label": "Finance / Accounting",
        "hard_skills": ["excel","financial modeling","accounting","sap","sql","forecasting","bookkeeping"],
        "soft_skills": ["attention to detail","communication","analytical thinking"],
        "edu_levels": ["bachelor","s1","master"]
    },
    "hr": {
        "label": "Human Resources",
        "hard_skills": ["hris","recruitment","onboarding","performance management"],
        "soft_skills": ["communication","empathy","conflict resolution"],
        "edu_levels": ["bachelor","s1"]
    },
    "design": {
        "label": "Design / Creative",
        "hard_skills": ["photoshop","illustrator","figma","ui ux","after effects","prototyping"],
        "soft_skills": ["creativity","collaboration","visual communication"],
        "edu_levels": ["bachelor","s1"]
    },
    "default": {
        "label": "General",
        "hard_skills": ["ms office","communication","administration"],
        "soft_skills": ["communication","teamwork"],
        "edu_levels": ["bachelor","s1"]
    }
}

# -------------------------
# Domain detection (simple overlap)
# -------------------------
def detect_domain(jd_text):
    jd_words = set(re.findall(r"\w+", jd_text.lower()))
    scores = {}
    for key, v in ESCO_TAXONOMY.items():
        hard = set(v["hard_skills"])
        overlap = len(hard & jd_words)
        scores[key] = overlap
    # pick best match (if all zero, default)
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] == 0:
        return "default"
    return best[0]

# -------------------------
# Embedding / similarity helper
# -------------------------
embedder = None
if USE_FASTEMBED:
    try:
        embedder = TextEmbedding()
    except Exception:
        embedder = None
        USE_FASTEMBED = False

tfidf_vectorizer = None

def semantic_similarity(a_text, b_text):
    """
    returns similarity 0..1
    uses fastembed if available, else TF-IDF, else token-overlap
    """
    if USE_FASTEMBED and embedder is not None:
        try:
            a_vec = np.array(list(embedder.embed([a_text]))[0])
            b_vec = np.array(list(embedder.embed([b_text]))[0])
            denom = (np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
            return float(np.dot(a_vec, b_vec) / denom) if denom > 0 else 0.0
        except Exception:
            pass

    # TF-IDF fallback
    if USE_TFIDF:
        global tfidf_vectorizer
        try:
            tfidf_vectorizer = TfidfVectorizer(max_features=15000, stop_words='english')
            vecs = tfidf_vectorizer.fit_transform([a_text, b_text])
            sim = sklearn_cosine(vecs[0], vecs[1])[0][0]
            return float(sim)
        except Exception:
            pass

    # token overlap fallback
    a_words = set(re.findall(r"\w+", a_text.lower()))
    b_words = set(re.findall(r"\w+", b_text.lower()))
    if not b_words:
        return 0.0
    return float(len(a_words & b_words) / len(b_words))

# -------------------------
# Keyword presence and ATS overlap
# -------------------------
def keyword_presence_score(text, keywords):
    t = text.lower()
    found = [k for k in keywords if k.lower() in t]
    missing = [k for k in keywords if k.lower() not in t]
    score = (len(found) / max(1, len(keywords))) * 100
    return score, found, missing

def ats_overlap_score(resume_text, jd_text):
    resume_words = set(re.findall(r"\w+", resume_text.lower()))
    jd_words = set(re.findall(r"\w+", jd_text.lower()))
    if not jd_words:
        return 0.0, []
    overlap = resume_words & jd_words
    score = (len(overlap) / len(jd_words)) * 100
    return score, list(overlap)

# -------------------------
# Weight presets per domain (example)
# -------------------------
DOMAIN_WEIGHTS = {
    "data_science": {"hard":0.40,"exp":0.25,"ats":0.15,"soft":0.10,"edu":0.10},
    "software_engineer": {"hard":0.45,"exp":0.25,"ats":0.15,"soft":0.10,"edu":0.05},
    "product_marketing": {"hard":0.30,"exp":0.25,"ats":0.20,"soft":0.15,"edu":0.10},
    "finance": {"hard":0.35,"exp":0.30,"ats":0.15,"soft":0.10,"edu":0.10},
    "hr": {"hard":0.20,"exp":0.25,"ats":0.20,"soft":0.25,"edu":0.10},
    "design": {"hard":0.30,"exp":0.20,"ats":0.10,"soft":0.30,"edu":0.10},
    "default": {"hard":0.30,"exp":0.25,"ats":0.20,"soft":0.15,"edu":0.10}
}

# -------------------------
# Refs (brief)
# -------------------------
REFS = {
    "hard": "Schmidt & Hunter (1998); MDPI Resume2Vec (2024) — hard skills strongly predict technical task performance.",
    "exp": "Person–Job Fit literature (e.g., PJFNN, arXiv 2018) — relevant experience correlates with adaptation & effectiveness.",
    "ats": "ATS practices & patents (keyword overlap) — keywords used as initial screening features.",
    "soft": "Robles (2012) & soft skill literature — soft skills are critical for employability but harder to detect from CV.",
    "edu": "Studies on education relevance (HR literature) — education level used as filter, lower predictive power than skills."
}

# -------------------------
# Compute all indicator scores (domain adaptive)
# -------------------------
def compute_scores(resume_text, jd_text, domain_key):
    domain = ESCO_TAXONOMY.get(domain_key, ESCO_TAXONOMY["default"])
    weights = DOMAIN_WEIGHTS.get(domain_key, DOMAIN_WEIGHTS["default"])

    # stage 1: keyword detection per domain hard/soft/edu
    hard_score_raw, hard_found, hard_missing = keyword_presence_score(resume_text, domain["hard_skills"])
    soft_score_raw, soft_found, soft_missing = keyword_presence_score(resume_text, domain["soft_skills"])
    edu_score_raw, edu_found, edu_missing = keyword_presence_score(resume_text, domain["edu_levels"])

    # experience: presence of experience tokens + semantic similarity of experience lines
    exp_score_raw, exp_found, exp_missing = keyword_presence_score(resume_text, ["years","year","experience","pengalaman","project","lead","managed"])

    # ATS overlap
    ats_score_raw, ats_overlap = ats_overlap_score(resume_text, jd_text)

    # semantic similarity overall
    sem_sim = semantic_similarity(resume_text, jd_text)  # 0..1

    # Combine rules (blend keyword % and semantic sim)
    hard_combined = 0.6 * hard_score_raw + 0.4 * (sem_sim * 100)
    exp_combined = 0.7 * exp_score_raw + 0.3 * (sem_sim * 100)
    soft_combined = 0.6 * soft_score_raw + 0.4 * (sem_sim * 100)
    edu_combined = edu_score_raw  # small weight anyway
    ats_combined = 0.8 * ats_score_raw + 0.2 * (sem_sim * 100)

    final = (
        hard_combined * weights["hard"] +
        exp_combined * weights["exp"] +
        ats_combined * weights["ats"] +
        soft_combined * weights["soft"] +
        edu_combined * weights["edu"]
    )

    details = {
        "domain": domain_key,
        "domain_label": domain["label"],
        "semantic_similarity": sem_sim,
        "hard": {"raw":round(hard_score_raw,2), "combined":round(hard_combined,2), "found":hard_found, "missing":hard_missing},
        "exp": {"raw":round(exp_score_raw,2), "combined":round(exp_combined,2), "found":exp_found, "missing":exp_missing},
        "soft": {"raw":round(soft_score_raw,2), "combined":round(soft_combined,2), "found":soft_found, "missing":soft_missing},
        "edu": {"raw":round(edu_score_raw,2), "combined":round(edu_combined,2), "found":edu_found, "missing":edu_missing},
        "ats": {"raw":round(ats_score_raw,2), "combined":round(ats_combined,2), "overlap":ats_overlap}
    }

    return float(round(final,2)), details

# -------------------------
# Build explanations & suggestions
# -------------------------
def build_explanations(final_score, details, jd_text):
    ex = {}
    # Hard
    h = details["hard"]
    ex["Hard Skill Match"] = {
        "score": h["combined"],
        "why": f"Sistem menemukan {len(h['found'])} dari {len(ESCO_TAXONOMY[details['domain']]['hard_skills'])} hard-skill domain. Ditemukan: {h['found']}. Tidak ditemukan (contoh): {h['missing'][:6]}",
        "ref": REFS["hard"],
        "suggestion": suggest_hard_skills_improvement(h['missing'], details['domain'])
    }
    # Experience
    e = details["exp"]
    ex["Experience Fit"] = {
        "score": e["combined"],
        "why": f"Kata kunci pengalaman terdeteksi: {e['found']}. Resume akan lebih terlihat relevan jika durasi proyek & hasil (KPI) dituliskan secara eksplisit.",
        "ref": REFS["exp"],
        "suggestion": suggest_experience_snippets()
    }
    # ATS
    a = details["ats"]
    ex["Keyword / ATS Match"] = {
        "score": a["combined"],
        "why": f"{len(a['overlap'])} kata JD cocok dengan resume. Contoh overlap (sample): {a['overlap'][:20]}",
        "ref": REFS["ats"],
        "suggestion": suggest_ats_snippets(jd_text, a['overlap'])
    }
    # Soft
    s = details["soft"]
    ex["Soft Skill Relevance"] = {
        "score": s["combined"],
        "why": f"Kata soft skill yang ditemukan: {s['found']}. Soft skill lebih meyakinkan jika disertai contoh (mis. 'led X-person team to...').",
        "ref": REFS["soft"],
        "suggestion": suggest_softskill_snippets()
    }
    # Edu
    ed = details["edu"]
    ex["Education Fit"] = {
        "score": ed["combined"],
        "why": f"Kata terkait pendidikan ditemukan: {ed['found']}. Jika peran mensyaratkan jurusan tertentu, tunjukkan transkrip/sertifikasi relevan.",
        "ref": REFS["edu"],
        "suggestion": suggest_education_snippets()
    }
    ex["Final Summary"] = {
        "score": final_score,
        "why": f"Final score dihitung dari bobot domain '{details['domain_label']}' dan kombinasi keyword + semantic similarity.",
        "ref": "Gabungan referensi per-indikator."
    }
    return ex

# Suggestion helper functions
def suggest_hard_skills_improvement(missing_list, domain_key):
    if not missing_list:
        return "Hard skill sudah lengkap; tambahkan contoh proyek/hasil untuk setiap skill."
    suggestions = []
    domain_skills = ESCO_TAXONOMY.get(domain_key, ESCO_TAXONOMY["default"])["hard_skills"]
    for k in missing_list[:5]:
        suggestions.append(f"Tambahkan kata/skill '{k}' jika relevan. Contoh kalimat: 'Used {k} to achieve X% improvement in Y.'")
    return " ".join(suggestions)

def suggest_experience_snippets():
    return ("Contoh kalimat: 'Led a 3-person team to deliver X feature in 6 months, "
            "improving throughput by 25%.' atau 'Reduced processing time by 30% via optimization.'")

def suggest_ats_snippets(jd_text, overlap):
    # try to produce 2-3 keyword phrases from JD to add
    jd_words = re.findall(r"\w+", jd_text.lower())
    # pick some high-frequency words excluding stopwords heuristically
    freq = {}
    for w in jd_words:
        if len(w) < 3: continue
        freq[w] = freq.get(w,0)+1
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    top = [w for w,c in sorted_words[:10] if w not in overlap]
    suggestions = []
    for t in top[:5]:
        suggestions.append(f"Masukkan istilah '{t}' jika relevan (contoh: 'Experience with {t}').")
    if not suggestions:
        return "Periksa JD untuk istilah teknis spesifik, lalu sisipkan istilah tersebut di bagian skill atau ringkasan."
    return " ".join(suggestions)

def suggest_softskill_snippets():
    return ("Contoh: 'Strong communication skills demonstrated by presenting roadmap to stakeholders', "
            "'Collaborated with cross-functional teams to deliver ...'")

def suggest_education_snippets():
    return ("Jika relevan, tambahkan detail: degree, major, institution, graduation year, atau sertifikasi profesional.")

# -------------------------
# PDF generation (reportlab)
# -------------------------
def split_text_to_lines(text, max_chars=90):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def export_pdf_bytes(report_title, resume_name, final_score, explanations, details, jd_text):
    if not REPORTLAB_AVAILABLE:
        return None
    buf = BytesIO()
    width, height = letter = (612.0, 792.0)
    c = canvas.Canvas(buf, pagesize=letter)
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, report_title)
    y -= 24
    c.setFont("Helvetica", 9)
    c.drawString(50, y, f"Resume: {resume_name}")
    c.drawString(350, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 24
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Final Score: {final_score:.2f} / 100")
    y -= 20
    c.setFont("Helvetica-Bold", 11)
    for key in ["Hard Skill Match","Experience Fit","Keyword / ATS Match","Soft Skill Relevance","Education Fit"]:
        v = explanations[key]
        if y < 120:
            c.showPage()
            y = height - 50
        c.drawString(50, y, f"{key} — {v['score']}")
        y -= 14
        c.setFont("Helvetica", 9)
        for ln in split_text_to_lines(v['why'], 90):
            if y < 80:
                c.showPage(); y = height - 50
            c.drawString(60, y, ln)
            y -= 12
        # reference
        if y < 80:
            c.showPage(); y = height - 50
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(60, y, f"Ref: {v['ref']}")
        y -= 18
        c.setFont("Helvetica-Bold", 11)
    # JD excerpt
    if jd_text:
        if y < 140:
            c.showPage(); y = height - 50
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Job Description (excerpt):")
        y -= 14
        c.setFont("Helvetica", 9)
        for ln in split_text_to_lines(jd_text.strip()[:1000], 90):
            if y < 80:
                c.showPage(); y = height - 50
            c.drawString(60, y, ln)
            y -= 12
    c.save()
    buf.seek(0)
    return buf.read()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Resume Analyzer", layout="wide")
st.title("Resume Analyzer")
st.caption("Built and developed by **Yogi Bima Dwi Graha.M**")

st.write("Upload resume (PDF/DOCX) dan masukkan Job Description. Sistem otomatis mendeteksi domain, menyesuaikan bobot, memberikan skor indikator, alasan, dan rekomendasi perbaikan.")

col1, col2 = st.columns([1,2])
with col1:
    uploaded = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf","docx"])
with col2:
    jd_text = st.text_area("Paste Job Description / Role Summary", height=260)

analyze = st.button("Analyze")

if analyze:
    if not uploaded:
        st.error("Silakan upload resume (PDF/DOCX).")
    elif not jd_text or len(jd_text.strip())<20:
        st.error("Masukkan Job Description yang cukup (≥20 char).")
    else:
        with st.spinner("Memproses..."):
            resume_text = extract_text(uploaded)
            if not resume_text or len(resume_text.strip())<10:
                st.warning("Teks resume tampak kosong atau tidak dapat diekstrak dengan baik.")
            domain_key = detect_domain(jd_text)
            final_score, details = compute_scores(resume_text, jd_text, domain_key)
            explanations = build_explanations(final_score, details, jd_text)

        # Display header + domain info
        st.header("📊 Results")
        st.subheader(f"Detected domain: **{details['domain_label']}**")
        st.metric(label="Final Screening Score", value=f"{final_score:.2f} / 100")

        # left/right columns for details
        left, right = st.columns([1,1])
        with left:
            st.subheader("Indicator breakdown")
            for k in ["Hard Skill Match","Experience Fit","Keyword / ATS Match","Soft Skill Relevance","Education Fit"]:
                v = explanations[k]
                st.markdown(f"**{k} — {v['score']}**")
                st.write(v["why"])
                st.caption("Ref: " + v["ref"])
                st.write("Suggestion:", v["suggestion"])
                st.write("---")

        with right:
            st.subheader("Technical details")
            st.write("Semantic similarity (resume ↔ JD):", round(details["semantic_similarity"],4))
            st.write("Hard skills found:", details["hard"]["found"])
            st.write("Experience tokens found:", details["exp"]["found"])
            st.write("Soft skills found:", details["soft"]["found"])
            st.write("Education tokens found:", details["edu"]["found"])
            st.write("ATS overlap sample:", details["ats"]["overlap"][:50])

        # Recommendations summary
        st.subheader("✍️ Quick suggested phrases to add to resume")
        st.write("Hard skill suggestions:")
        st.write(explanations["Hard Skill Match"]["suggestion"])
        st.write("Experience suggestion:")
        st.write(explanations["Experience Fit"]["suggestion"])
        st.write("Soft skill suggestion:")
        st.write(explanations["Soft Skill Relevance"]["suggestion"])
        st.write("Education suggestion:")
        st.write(explanations["Education Fit"]["suggestion"])

        # Export PDF
        st.subheader("📥 Export Report")
        pdf_bytes = export_pdf_bytes(
            report_title="Resume Screening Report",
            resume_name=uploaded.name,
            final_score=final_score,
            explanations=explanations,
            details=details,
            jd_text=jd_text
        )
        if pdf_bytes is None:
            st.warning("Reportlab tidak terinstall — PDF export tidak tersedia. Install 'reportlab' untuk mengaktifkan PDF export.")
        else:
            st.download_button("Download PDF report", data=pdf_bytes,
                               file_name=f"screening_report_{os.path.splitext(uploaded.name)[0]}.pdf",
                               mime="application/pdf")

        st.success("Selesai — periksa hasil dan rekomendasi di atas.")
