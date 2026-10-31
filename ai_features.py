"""Free hybrid AI features for ResumeIQ: deterministic NLP + optional local Ollama."""
import json
import os
import re
import urllib.request
import urllib.error
from collections import Counter

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

SKILL_TERMS = [
    "python", "java", "c++", "javascript", "typescript", "sql", "mysql", "postgresql", "mongodb",
    "flask", "django", "fastapi", "react", "node.js", "html", "css", "git", "github", "linux",
    "data structures", "algorithms", "oop", "dbms", "operating systems", "computer networks",
    "machine learning", "deep learning", "artificial intelligence", "nlp", "computer vision", "tensorflow",
    "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib", "opencv", "power bi", "tableau",
    "excel", "statistics", "data analysis", "aws", "azure", "docker", "kubernetes", "rest api", "streamlit",
    "communication", "leadership", "problem solving", "project management", "generative ai", "llm", "fastapi"
]


def _unique(items):
    return list(dict.fromkeys(items))


def extract_skills(text):
    low = (text or "").lower()
    return [s.title() if s not in {"c++", "sql", "nlp", "aws", "azure", "oop", "dbms", "llm", "api"} else s.upper()
            for s in SKILL_TERMS if re.search(r"(?<![\w])" + re.escape(s) + r"(?![\w])", low)]


def _ollama(prompt):
    """Return local model output if Ollama is running; otherwise None (no API key)."""
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            data = json.loads(response.read().decode("utf-8"))
            return (data.get("response") or "").strip() or None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None


def generate(prompt, fallback):
    result = _ollama(prompt) if os.getenv("RESUMEIQ_USE_OLLAMA", "true").lower() in {"1", "true", "yes"} else None
    return {"text": result or fallback, "engine": "Ollama local LLM" if result else "Python NLP fallback"}


def analyze_resume(resume_text):
    text = (resume_text or "").strip()
    skills = extract_skills(text)
    sections = {name: bool(re.search(pattern, text, re.I)) for name, pattern in {
        "Education": r"\beducation\b|\bacademic\b", "Experience": r"\bexperience\b|\binternship\b",
        "Projects": r"\bprojects?\b", "Certifications": r"\bcertifications?\b|\bcourses?\b",
        "Contact details": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
    }.items()}
    words = re.findall(r"\b\w+\b", text)
    tips = []
    if len(words) < 180: tips.append("Add more relevant detail; the resume text is quite short.")
    if not sections["Projects"]: tips.append("Add 2–3 projects with your contribution, technology and measurable result.")
    if not sections["Experience"]: tips.append("Add internships, volunteering, freelance work or practical experience if available.")
    if not sections["Certifications"]: tips.append("Add relevant certifications or completed courses, if you have them.")
    if not re.search(r"\b(improved|built|developed|reduced|increased|automated|created|implemented)\b", text, re.I):
        tips.append("Use action verbs and quantify results where truthful (e.g. reduced processing time by 20%).")
    score = min(100, max(0, 35 + min(len(skills) * 4, 28) + sum(sections.values()) * 5 + (10 if len(words) >= 250 else 0)))
    fallback = f"Resume snapshot: {len(words)} words, {len(skills)} recognized skills. Sections found: {', '.join(k for k,v in sections.items() if v) or 'none detected'}.\n\nPriority improvements:\n" + "\n".join(f"• {x}" for x in (tips or ["Tailor the resume to each job description and keep achievements specific."]))
    prompt = "Review this resume for a student/job applicant. Give concise, evidence-based strengths, gaps, and practical improvements. Do not invent qualifications.\nRESUME:\n" + text[:9000]
    ai = generate(prompt, fallback)
    return {"score": score, "word_count": len(words), "skills": skills, "sections": sections, "tips": tips, **ai}


def match_job(resume_text, job_description):
    rs, js = set(s.lower() for s in extract_skills(resume_text)), set(s.lower() for s in extract_skills(job_description))
    matched, missing = sorted(rs & js), sorted(js - rs)
    pct = round(100 * len(matched) / len(js)) if js else 0
    fallback = f"Keyword match: {pct}% ({len(matched)} of {len(js)} detected job skills).\nMatched: {', '.join(matched) or 'None detected'}\nMissing: {', '.join(missing) or 'No missing detected skills'}\n\nUse missing skills as a learning checklist; keyword overlap is not a hiring prediction."
    prompt = f"Compare resume to job description. Identify matched strengths and gaps; never claim a missing skill is possessed.\nResume:\n{resume_text[:5000]}\nJob description:\n{job_description[:3500]}"
    ai = generate(prompt, fallback)
    return {"percentage": pct, "matched": matched, "missing": missing, **ai}


def improve_resume(resume_text):
    fallback = "Rewrite tips:\n• Start bullets with strong verbs (Built, Designed, Automated, Analysed).\n• Use: Action + task + technology + result.\n• Replace generic claims such as ‘hardworking’ with proof from projects.\n• Keep dates, links and formatting consistent.\n• Never add skills or metrics that are not true.\n\nExample template:\n‘Developed [project] using [technology], implementing [feature] and achieving [measurable result].’"
    prompt = "Improve the wording and structure of this resume while preserving only facts already present. Clearly label any missing information as placeholders; do not invent metrics. Return a concise improved draft and prioritized changes.\n" + (resume_text or "")[:9000]
    return generate(prompt, fallback)


def interview_prep(resume_text, job_description=""):
    skills = extract_skills((resume_text or "") + " " + (job_description or ""))
    questions = ["Walk me through your resume and the role you are targeting.", "Describe a project you built: problem, your contribution, tools and result.", "Tell me about a bug or difficult problem you solved."]
    questions += [f"Explain {s} and describe a practical example of using it." for s in skills[:5]]
    questions += ["What would you improve if you had one more week on your strongest project?", "What are you currently learning for this role?"]
    fallback = "Practice questions:\n" + "\n".join(f"{i+1}. {q}" for i,q in enumerate(questions)) + "\n\nAnswer with Situation → Task → Action → Result. Keep answers truthful and specific."
    prompt = f"Create 8 interview questions grounded in this resume and job description, with short answer guidance. Do not invent candidate experience.\nResume:\n{resume_text[:5000]}\nJD:\n{job_description[:2000]}"
    return generate(prompt, fallback)


def learning_roadmap(resume_text, job_description=""):
    missing = sorted(set(s.lower() for s in extract_skills(job_description)) - set(s.lower() for s in extract_skills(resume_text)))
    focus = ", ".join(missing[:5]) or "core role skills and project depth"
    fallback = ("6-week roadmap (adjust to your schedule):\n"
      "Week 1: Review target job descriptions; audit fundamentals and set a baseline.\n"
      f"Week 2: Learn/practice: {focus}.\nWeek 3: Build a small, demonstrable project using one priority skill.\n"
      "Week 4: Add tests, documentation, Git commits and a clear README.\n"
      "Week 5: Practice 2 coding/technical questions daily and explain your project aloud.\n"
      "Week 6: Tailor resume, update portfolio, run mock interviews and apply to suitable roles.\n\n"
      "Weekly habit: track progress and only list skills you can demonstrate.")
    prompt = f"Create a practical 6-week learning roadmap based on the resume and target job. Prioritize gaps, free resources/topics and a small portfolio outcome each week.\nResume:\n{resume_text[:4500]}\nJob description:\n{job_description[:2500]}"
    return generate(prompt, fallback)
