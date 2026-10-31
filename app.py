from flask import Flask, render_template, request, send_file, redirect, url_for, session, flash
import os
from dotenv import load_dotenv
import re
import io
import json
from datetime import datetime

from pypdf import PdfReader
from docx import Document

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from xml.sax.saxutils import escape

from database.skills import SKILLS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from ai_features import analyze_resume as ai_analyze_resume, match_job as ai_match_job, improve_resume as ai_improve_resume, interview_prep as ai_interview_prep, learning_roadmap as ai_learning_roadmap

# Load environment variables from the project .env file.
load_dotenv()


# =========================================================
# LOCAL PROCESSING
# =========================================================

# This project works without an external AI API.
# Resume analysis, summary generation and interview questions
# are generated locally from the extracted resume information.

# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

# Authentication and local SQLite database configuration.
# Set SECRET_KEY in your .env file before deploying publicly.
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
if not app.config["SECRET_KEY"]:
    raise RuntimeError(
        "SECRET_KEY is missing. Add SECRET_KEY=<your generated key> to the .env file."
    )
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("SESSION_COOKIE_SECURE", "False").lower() == "true"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///resume_analyzer.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)


class ResumeAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    candidate_name = db.Column(db.String(150), default="Not detected")
    ats_score = db.Column(db.Integer, nullable=False, default=0)
    job_match_percentage = db.Column(db.Integer, nullable=True)
    skills_json = db.Column(db.Text, nullable=False, default="[]")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user = db.relationship("User", backref=db.backref("resume_analyses", lazy=True))


with app.app_context():
    db.create_all()


def login_required(view_function):
    """Redirect visitors to the login page before protected features."""
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to use this feature.", "warning")
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)
    return wrapped_view

UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB maximum

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# ALLOWED FILE TYPES
# =========================================================

ALLOWED_EXTENSIONS = {"pdf", "docx"}


def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(file_path):

    text = ""

    try:

        reader = PdfReader(file_path)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:

        print("PDF extraction error:", e)

    return text


# =========================================================
# DOCX TEXT EXTRACTION
# =========================================================

def extract_docx_text(file_path):

    text = ""

    try:

        document = Document(file_path)

        for paragraph in document.paragraphs:

            if paragraph.text.strip():

                text += paragraph.text + "\n"

    except Exception as e:

        print("DOCX extraction error:", e)

    return text


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n+", "\n", text)

    return text.strip()


# =========================================================
# EMAIL EXTRACTION
# =========================================================

def extract_email(text):

    pattern = (
        r"[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    )

    match = re.search(pattern, text)

    if match:
        return match.group(0)

    return "Not found"


# =========================================================
# PHONE EXTRACTION
# =========================================================

def extract_phone(text):

    pattern = r"(?:\+91[\s-]?)?[6-9]\d{9}"

    match = re.search(pattern, text)

    if match:
        return match.group(0)

    return "Not found"


# =========================================================
# NAME EXTRACTION
# =========================================================

def extract_name(text):
    """Conservatively detect a likely candidate name near the top of the resume."""
    if not text:
        return "Not found"
    invalid_headings = {
        "CAREER OBJECTIVE", "OBJECTIVE", "RESUME", "CURRICULUM VITAE", "PERSONAL INFORMATION",
        "CONTACT", "CONTACT INFORMATION", "EDUCATION", "EXPERIENCE", "WORK EXPERIENCE",
        "PROFESSIONAL EXPERIENCE", "SKILLS", "TECHNICAL SKILLS", "PROJECTS", "CERTIFICATIONS",
        "ACHIEVEMENTS", "SUMMARY", "PROFILE", "CAREER SUMMARY", "INTELLIGENCE AND MACHINE LEARNING"
    }
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:15]:
        cleaned = re.sub(r"^[^A-Za-z]+|[^A-Za-z.\-' ]+$", "", line).strip()
        normalized = re.sub(r"\s+", " ", cleaned).upper()
        if not cleaned or normalized in invalid_headings:
            continue
        if any(x in normalized for x in ("CAREER OBJECTIVE", "PERSONAL INFORMATION", "CURRICULUM VITAE", "WORK EXPERIENCE", "TECHNICAL SKILLS")):
            continue
        if "@" in line or re.search(r"\d", line) or len(cleaned) > 55:
            continue
        if any(x in normalized for x in ("EMAIL", "PHONE", "MOBILE", "LINKEDIN", "GITHUB", "ADDRESS")):
            continue
        words = cleaned.split()
        if 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Za-z][A-Za-z.\-' ]*", word) for word in words):
            return cleaned
    return "Not found"


# =========================================================
# RESUME SKILL DETECTION
# =========================================================

def detect_skills(text):

    detected_skills = []

    text_lower = text.lower()

    for skill in SKILLS:

        if skill.lower() in text_lower:

            detected_skills.append(skill)

    return detected_skills


# =========================================================
# JOB DESCRIPTION SKILL DETECTION
# =========================================================

def detect_jd_skills(job_description):

    detected_skills = []

    if not job_description:
        return detected_skills

    jd_lower = job_description.lower()

    for skill in SKILLS:

        if skill.lower() in jd_lower:

            detected_skills.append(skill)

    return detected_skills


# =========================================================
# JOB MATCH CALCULATION
# =========================================================

def calculate_job_match(
    resume_skills,
    jd_skills
):

    if not jd_skills:

        return 0, [], []

    resume_skill_set = {
        skill.lower()
        for skill in resume_skills
    }

    matched_skills = []

    missing_skills = []

    for skill in jd_skills:

        if skill.lower() in resume_skill_set:

            matched_skills.append(skill)

        else:

            missing_skills.append(skill)

    match_percentage = round(
        (
            len(matched_skills)
            / len(jd_skills)
        ) * 100
    )

    return (
        match_percentage,
        matched_skills,
        missing_skills
    )


# =========================================================
# MISSING SKILL RECOMMENDATIONS
# =========================================================

def generate_missing_skill_recommendations(
    missing_skills
):

    recommendations = []

    if not missing_skills:

        recommendations.append(
            "Excellent! No major technical skill gaps were detected."
        )

        return recommendations

    learning_resources = {

        "Python":
            "Learn Python fundamentals, OOP, modules and practical projects.",

        "Java":
            "Learn Java OOP, collections, exception handling and JDBC.",

        "C++":
            "Learn C++ fundamentals, OOP, STL and problem solving.",

        "C":
            "Learn C programming, pointers, arrays, functions and memory management.",

        "JavaScript":
            "Learn JavaScript fundamentals, DOM, ES6 and asynchronous programming.",

        "TypeScript":
            "Learn TypeScript types, interfaces, classes and modern JavaScript development.",

        "HTML":
            "Learn semantic HTML and proper webpage structure.",

        "CSS":
            "Learn CSS layouts, Flexbox, Grid and responsive design.",

        "React":
            "Learn React components, props, state, hooks and API integration.",

        "Angular":
            "Learn Angular components, services, routing and TypeScript.",

        "Vue":
            "Learn Vue components, reactivity and application structure.",

        "Node.js":
            "Learn Node.js, Express, REST APIs and backend development.",

        "Express.js":
            "Learn Express routing, middleware and REST API development.",

        "Flask":
            "Learn Flask routing, templates, forms, APIs and database integration.",

        "Django":
            "Learn Django models, views, templates, forms and REST APIs.",

        "FastAPI":
            "Learn FastAPI, REST APIs, validation and asynchronous programming.",

        "SQL":
            "Learn SQL queries, joins, subqueries, aggregation and window functions.",

        "MySQL":
            "Learn MySQL databases, queries, joins, indexes and normalization.",

        "PostgreSQL":
            "Learn PostgreSQL queries, database design and advanced SQL.",

        "MongoDB":
            "Learn MongoDB collections, documents, queries and aggregation.",

        "NumPy":
            "Learn NumPy arrays, indexing, vectorization and numerical operations.",

        "Pandas":
            "Learn Pandas DataFrames, data cleaning, filtering and analysis.",

        "Matplotlib":
            "Learn Matplotlib charts, plots and data visualization.",

        "Scikit-learn":
            "Learn machine learning algorithms, preprocessing, training and evaluation.",

        "Machine Learning":
            "Learn supervised learning, unsupervised learning, feature engineering and model evaluation.",

        "Deep Learning":
            "Learn neural networks, CNNs, RNNs, optimization and deep learning frameworks.",

        "NLP":
            "Learn text preprocessing, embeddings, classification and transformer basics.",

        "Computer Vision":
            "Learn image processing, OpenCV, feature extraction and computer vision models.",

        "Generative AI":
            "Learn LLM fundamentals, prompt engineering, embeddings, RAG and AI APIs.",

        "Artificial Intelligence":
            "Learn search algorithms, knowledge representation, machine learning and intelligent agents.",

        "TensorFlow":
            "Learn TensorFlow tensors, neural networks, training and model deployment.",

        "Keras":
            "Learn Keras models, layers, training and neural network workflows.",

        "PyTorch":
            "Learn tensors, datasets, neural networks and model training with PyTorch.",

        "OpenCV":
            "Learn image processing, object detection and computer vision using OpenCV.",

        "MediaPipe":
            "Learn MediaPipe pipelines and real-time computer vision applications.",

        "Git":
            "Learn Git commands, branching, merging, commits and collaboration.",

        "GitHub":
            "Learn repositories, pull requests, branches and collaborative development.",

        "Docker":
            "Learn Docker images, containers, Dockerfiles and deployment basics.",

        "Linux":
            "Learn Linux commands, shell scripting, permissions and process management.",

        "AWS":
            "Learn AWS fundamentals, EC2, S3, IAM and basic cloud deployment.",

        "Microsoft Azure":
            "Learn Azure fundamentals, compute, storage and cloud deployment.",

        "Google Cloud":
            "Learn Google Cloud fundamentals, compute, storage and deployment.",

        "Data Structures":
            "Practice arrays, strings, linked lists, stacks, queues, trees, graphs and hash tables.",

        "Algorithms":
            "Learn searching, sorting, recursion, greedy algorithms, dynamic programming and graph algorithms.",

        "DBMS":
            "Learn SQL, normalization, transactions, indexing and database design.",

        "Operating Systems":
            "Learn processes, threads, scheduling, memory management and file systems.",

        "Computer Networks":
            "Learn OSI/TCP-IP models, HTTP, DNS, TCP, UDP and networking fundamentals.",

        "Object Oriented Programming":
            "Learn classes, objects, inheritance, polymorphism, abstraction and encapsulation.",

        "Software Engineering":
            "Learn SDLC, requirements engineering, design, testing and software development methodologies."
    }

    for skill in missing_skills:

        if skill in learning_resources:

            recommendations.append(
                f"{skill}: {learning_resources[skill]}"
            )

        else:

            recommendations.append(
                f"{skill}: Add this skill to your learning plan "
                f"and practice it through a small project."
            )

    return recommendations


# =========================================================
# JOB-SPECIFIC RESUME IMPROVEMENT
# =========================================================

def generate_job_improvement_suggestions(
    job_description,
    resume_skills,
    matched_skills,
    missing_skills,
    projects,
    experience
):

    suggestions = []

    if not job_description:
        return suggestions

    if missing_skills:
        suggestions.append(
            "Prioritize the missing job skills: "
            + ", ".join(missing_skills)
            + ". Add them only if you genuinely know or have practiced them."
        )

    if matched_skills:
        suggestions.append(
            "Keep the matched skills visible in the resume and connect them "
            "to projects, coursework, internships, or practical work."
        )

    if projects == "Not found":
        suggestions.append(
            "Add at least one project relevant to the job description and "
            "mention the technologies and your contribution."
        )

    if experience == "Not found":
        suggestions.append(
            "If applicable, add internships, training, freelance work, or "
            "practical experience related to the target role."
        )

    if not suggestions:
        suggestions.append(
            "The detected resume skills already align with the job description. "
            "Tailor the wording of your projects and experience to the role."
        )

    return suggestions


# =========================================================
# SECTION EXTRACTION
# =========================================================

def extract_section(
    text,
    section_keywords
):

    lines = text.split("\n")

    collecting = False

    section_lines = []

    all_sections = [
        "education",
        "experience",
        "work experience",
        "projects",
        "certifications",
        "skills",
        "technical skills",
        "summary",
        "objective",
        "achievements",
        "contact",
        "personal information"
    ]

    for line in lines:

        clean_line = line.strip()

        if not clean_line:
            continue

        lower_line = clean_line.lower()

        if any(
            keyword in lower_line
            for keyword in section_keywords
        ):

            collecting = True

            continue

        if collecting:

            if any(
                section in lower_line
                for section in all_sections
                if section not in section_keywords
            ):

                break

            section_lines.append(
                clean_line
            )

    if section_lines:

        return "\n".join(section_lines)

    return "Not found"


# =========================================================
# RESUME QUALITY
# =========================================================

def analyze_resume_quality(
    name,
    email,
    phone,
    skills,
    education,
    experience,
    projects
):

    issues = []

    if name == "Not found":
        issues.append("Name missing")

    if email == "Not found":
        issues.append("Email missing")

    if phone == "Not found":
        issues.append("Phone number missing")

    if len(skills) < 5:
        issues.append("Few technical skills detected")

    if education == "Not found":
        issues.append("Education section missing")

    if experience == "Not found":
        issues.append("Experience section missing")

    if projects == "Not found":
        issues.append("Projects section missing")

    if not issues:
        return "Good resume structure."

    return (
        "Needs improvement: "
        + ", ".join(issues)
    )


# =========================================================
# ATS SCORE
# =========================================================

def calculate_ats_score(
    name,
    email,
    phone,
    skills,
    education,
    experience,
    projects,
    certifications
):

    score = 0

    if name != "Not found":
        score += 10

    if email != "Not found":
        score += 10

    if phone != "Not found":
        score += 10

    if len(skills) >= 5:
        score += 20

    elif len(skills) >= 3:
        score += 10

    if education != "Not found":
        score += 10

    if experience != "Not found":
        score += 15

    if projects != "Not found":
        score += 15

    if certifications != "Not found":
        score += 10

    return min(score, 100)


# =========================================================
# RESUME ANALYTICS DASHBOARD
# =========================================================

def generate_resume_dashboard(
    ats_score,
    job_description,
    job_match_percentage,
    skills,
    education,
    experience,
    projects,
    certifications
):

    sections = {
        "Education": education != "Not found",
        "Experience": experience != "Not found",
        "Projects": projects != "Not found",
        "Certifications": certifications != "Not found"
    }

    completed_sections = sum(sections.values())

    if job_description and job_description.strip():
        job_match_display = f"{job_match_percentage}%"
    else:
        job_match_display = "N/A"

    return {
        "ats_score": ats_score,
        "job_match_percentage": job_match_percentage,
        "job_match_display": job_match_display,
        "skills_count": len(skills),
        "completed_sections": completed_sections,
        "total_sections": len(sections),
        "sections": sections
    }


# =========================================================
# ATS SCORE IMPROVEMENT ANALYSIS
# =========================================================

def generate_ats_improvement(
    name,
    email,
    phone,
    skills,
    education,
    experience,
    projects,
    certifications,
    resume_text,
    ats_score
):

    improvements = []

    if name == "Not found":
        improvements.append(
            "Add your full name clearly at the top of the resume."
        )

    if email == "Not found":
        improvements.append(
            "Add a professional email address."
        )

    if phone == "Not found":
        improvements.append(
            "Add a valid phone number."
        )

    if len(skills) < 5:
        improvements.append(
            "Add more job-relevant technical skills and mention them "
            "in the context of your projects or experience."
        )

    if education == "Not found":
        improvements.append(
            "Add a clearly labelled Education section with degree, "
            "college, and relevant academic details."
        )

    if experience == "Not found":
        improvements.append(
            "Add internship, training, freelance, or practical "
            "experience when available."
        )

    if projects == "Not found":
        improvements.append(
            "Add 2–3 relevant projects with technologies used, "
            "your contribution, and measurable outcomes."
        )

    if certifications == "Not found":
        improvements.append(
            "Add relevant certifications or courses if you have completed them."
        )

    word_count = len(resume_text.split())

    if word_count < 200:
        improvements.append(
            "Add meaningful details about projects, skills, education, "
            "and experience so the resume contains stronger searchable content."
        )

    if ats_score < 60:
        target_message = (
            "Your resume is below 60/100. Focus first on contact details, "
            "skills, education, projects, and relevant experience."
        )
    elif ats_score < 75:
        target_message = (
            "Your resume is in the developing range. Add missing sections "
            "and more job-relevant keywords to move toward 75+."
        )
    elif ats_score < 85:
        target_message = (
            "Your resume is good. Tailor keywords and strengthen project "
            "and experience descriptions to move toward 85+."
        )
    else:
        target_message = (
            "Your resume has strong basic ATS coverage. Tailor it to each "
            "job description and keep the formatting simple and consistent."
        )

    if not improvements:
        improvements.append(
            "No major structural improvement was detected from the current checks."
        )

    return {
        "current_score": ats_score,
        "target_score": 85,
        "target_message": target_message,
        "improvements": improvements
    }


# =========================================================
# STRENGTHS
# =========================================================

def generate_strengths(
    name,
    email,
    phone,
    skills,
    education,
    experience,
    projects,
    certifications,
    ats_score
):

    strengths = []

    if name != "Not found":
        strengths.append(
            "Name is clearly available in the resume."
        )

    if email != "Not found":
        strengths.append(
            "Professional email address is available."
        )

    if phone != "Not found":
        strengths.append(
            "Contact phone number is available."
        )

    if len(skills) >= 7:
        strengths.append(
            "Good number of technical skills detected."
        )

    elif len(skills) >= 4:
        strengths.append(
            "A reasonable set of technical skills is present."
        )

    if education != "Not found":
        strengths.append(
            "Education section is present."
        )

    if experience != "Not found":
        strengths.append(
            "Experience section is present."
        )

    if projects != "Not found":
        strengths.append(
            "Projects section is present."
        )

    if certifications != "Not found":
        strengths.append(
            "Certifications section is present."
        )

    if ats_score >= 80:
        strengths.append(
            "Overall ATS compatibility is strong."
        )

    elif ats_score >= 70:
        strengths.append(
            "Overall ATS compatibility is good."
        )

    return strengths


# =========================================================
# WEAKNESSES
# =========================================================

def generate_weaknesses(
    name,
    email,
    phone,
    skills,
    education,
    experience,
    projects,
    certifications,
    resume_text,
    ats_score
):

    weaknesses = []

    if name == "Not found":
        weaknesses.append(
            "Name could not be detected."
        )

    if email == "Not found":
        weaknesses.append(
            "Email address is missing or could not be detected."
        )

    if phone == "Not found":
        weaknesses.append(
            "Phone number is missing or could not be detected."
        )

    if len(skills) < 5:
        weaknesses.append(
            "The resume contains relatively few detected technical skills."
        )

    if education == "Not found":
        weaknesses.append(
            "Education section is missing."
        )

    if experience == "Not found":
        weaknesses.append(
            "Experience section is missing."
        )

    if projects == "Not found":
        weaknesses.append(
            "Projects section is missing."
        )

    if certifications == "Not found":
        weaknesses.append(
            "Certifications section is missing."
        )

    word_count = len(
        resume_text.split()
    )

    if word_count < 200:
        weaknesses.append(
            "Resume contains relatively little textual content."
        )

    if ats_score < 60:
        weaknesses.append(
            "Overall ATS compatibility needs significant improvement."
        )

    return weaknesses


# =========================================================
# GENERAL RECOMMENDATIONS
# =========================================================

def generate_recommendations(
    skills,
    education,
    experience,
    projects,
    certifications,
    resume_text,
    ats_score
):

    recommendations = []

    if len(skills) < 5:
        recommendations.append(
            "Add more relevant technical skills that match your target job."
        )

    if education == "Not found":
        recommendations.append(
            "Add a clearly labelled Education section."
        )

    if experience == "Not found":
        recommendations.append(
            "Add internships, work experience, or relevant practical experience."
        )

    if projects == "Not found":
        recommendations.append(
            "Add 2–3 relevant projects with technologies and measurable results."
        )

    if certifications == "Not found":
        recommendations.append(
            "Add relevant certifications or courses if available."
        )

    word_count = len(
        resume_text.split()
    )

    if word_count < 200:
        recommendations.append(
            "Add more meaningful details about projects, skills, education, and experience."
        )

    if ats_score < 70:
        recommendations.append(
            "Improve the resume structure and increase job-relevant keywords."
        )

    if not recommendations:
        recommendations.append(
            "Resume structure looks good. Focus on tailoring keywords for each job."
        )

    return recommendations


# =========================================================
# LOCAL RESUME SUMMARY
# =========================================================

def generate_ai_resume_summary(
    resume_text,
    name,
    skills,
    education,
    experience,
    projects
):

    if not resume_text:
        return "No resume content available for analysis."

    candidate_name = (
        name if name and name != "Not found"
        else "The candidate"
    )

    summary_parts = []

    if skills:
        skill_text = ", ".join(skills[:8])
        summary_parts.append(
            f"{candidate_name} has demonstrated skills in {skill_text}."
        )
    else:
        summary_parts.append(
            f"{candidate_name} has submitted a resume with limited detectable technical skills."
        )

    if education != "Not found":
        summary_parts.append(
            "The resume contains an education section describing the candidate's academic background."
        )

    if experience != "Not found":
        summary_parts.append(
            "The resume includes experience or internship information relevant to professional development."
        )

    if projects != "Not found":
        summary_parts.append(
            "The candidate has included project work that can be discussed during technical interviews."
        )

    if not skills and education == "Not found" and experience == "Not found" and projects == "Not found":
        summary_parts.append(
            "Adding clearly labelled skills, education, projects and experience details would make the resume easier to evaluate."
        )

    return " ".join(summary_parts[:5])


# =========================================================
# LOCAL INTERVIEW QUESTIONS
# =========================================================

def generate_ai_interview_questions(
    resume_text,
    skills,
    projects,
    experience
):

    technical_question_bank = {
        "Python": [
            "What is the difference between a list, tuple and set in Python?",
            "Explain object-oriented programming in Python with an example.",
            "How does exception handling work in Python?",
            "What is the difference between == and is in Python?",
            "How would you improve the performance of a Python program?"
        ],
        "Java": [
            "Explain the four main principles of object-oriented programming in Java.",
            "What is the difference between an interface and an abstract class in Java?",
            "How does exception handling work in Java?",
            "What is the difference between ArrayList and LinkedList?",
            "Explain method overloading and method overriding."
        ],
        "C++": [
            "Explain the difference between a pointer and a reference in C++.",
            "What are constructors and destructors in C++?",
            "Explain inheritance and polymorphism in C++.",
            "What is the STL and why is it useful?",
            "What is the difference between stack and heap memory?"
        ],
        "C": [
            "What is a pointer in C and why is it used?",
            "Explain the difference between an array and a pointer in C.",
            "What is dynamic memory allocation in C?",
            "What is the difference between malloc and calloc?",
            "Explain the use of structures in C."
        ],
        "SQL": [
            "What is the difference between WHERE and HAVING in SQL?",
            "Explain INNER JOIN and LEFT JOIN with an example.",
            "What is database normalization and why is it useful?",
            "What is the difference between DELETE, DROP and TRUNCATE?",
            "How would you find duplicate records in a SQL table?"
        ],
        "MySQL": [
            "What is the difference between a primary key and a foreign key in MySQL?",
            "Explain different types of SQL joins.",
            "What is an index and how does it improve query performance?",
            "What is normalization in database design?",
            "How would you optimize a slow MySQL query?"
        ],
        "JavaScript": [
            "What is the difference between var, let and const?",
            "Explain the difference between == and === in JavaScript.",
            "What is the DOM and how is it used?",
            "What are promises and async/await?",
            "Explain event bubbling in JavaScript."
        ],
        "Flask": [
            "What is Flask and why would you choose it for a web application?",
            "Explain routing in Flask.",
            "How do templates work in Flask?",
            "How would you handle form data and file uploads in Flask?",
            "How can a Flask application connect to a database?"
        ],
        "Django": [
            "What is Django and what is the purpose of its MVT architecture?",
            "Explain Django models, views and templates.",
            "What are Django migrations?",
            "How does URL routing work in Django?",
            "How would you build a REST API using Django?"
        ],
        "NumPy": [
            "What is a NumPy array and how is it different from a Python list?",
            "Explain vectorization in NumPy.",
            "What is broadcasting in NumPy?",
            "How do you reshape a NumPy array?",
            "Why is NumPy useful for data science?"
        ],
        "Pandas": [
            "What is a Pandas DataFrame?",
            "How do you handle missing values in Pandas?",
            "What is the difference between loc and iloc?",
            "How do you filter rows in a DataFrame?",
            "How would you group and aggregate data using Pandas?"
        ],
        "Machine Learning": [
            "What is the difference between supervised and unsupervised learning?",
            "What is overfitting and how can it be reduced?",
            "Why do we split data into training and testing sets?",
            "What is the difference between classification and regression?",
            "Which metrics can be used to evaluate a classification model?"
        ],
        "Deep Learning": [
            "What is a neural network and how does it learn?",
            "What is the role of an activation function?",
            "What is backpropagation?",
            "What is overfitting in deep learning and how can it be reduced?",
            "What is the difference between CNN and RNN?"
        ],
        "Computer Vision": [
            "What is computer vision and where is it used?",
            "What is image preprocessing?",
            "Explain image classification and object detection.",
            "What is the purpose of feature extraction in computer vision?",
            "How would you build a basic computer vision application?"
        ],
        "OpenCV": [
            "What is OpenCV and why is it used?",
            "How do you read and display an image using OpenCV?",
            "What is image thresholding?",
            "Explain edge detection in OpenCV.",
            "How can OpenCV be used for real-time video processing?"
        ],
        "Data Structures": [
            "What is the difference between an array and a linked list?",
            "Explain stack and queue with practical examples.",
            "What is a hash table and what is its average lookup complexity?",
            "Explain binary search and its time complexity.",
            "What is a binary search tree?"
        ],
        "Algorithms": [
            "What is time complexity and why is it important?",
            "Explain the difference between BFS and DFS.",
            "What is the difference between greedy algorithms and dynamic programming?",
            "Explain the time complexity of merge sort.",
            "What is recursion and when would you use it?"
        ],
        "DBMS": [
            "What is DBMS and why is it used?",
            "Explain normalization and its normal forms.",
            "What are ACID properties?",
            "What is the difference between a primary key and a candidate key?",
            "What is a transaction in DBMS?"
        ],
        "Operating Systems": [
            "What is the difference between a process and a thread?",
            "Explain CPU scheduling.",
            "What is deadlock and what are its necessary conditions?",
            "Explain virtual memory.",
            "What is the difference between paging and segmentation?"
        ],
        "Computer Networks": [
            "Explain the OSI model.",
            "What is the difference between TCP and UDP?",
            "What happens when you enter a URL in a browser?",
            "What is DNS and why is it required?",
            "What is the difference between HTTP and HTTPS?"
        ],
        "Software Engineering": [
            "What is SDLC and why is it important?",
            "Explain the Waterfall model.",
            "What is the difference between functional and non-functional requirements?",
            "What is software testing and why is it important?",
            "Explain cohesion and coupling."
        ]
    }

    technical_questions = []

    # Prefer questions from skills actually detected in the resume.
    for skill in skills:
        if skill in technical_question_bank:
            for question in technical_question_bank[skill]:
                if question not in technical_questions:
                    technical_questions.append(question)
                if len(technical_questions) == 5:
                    break
        if len(technical_questions) == 5:
            break

    # Fallback questions for detected skills without a dedicated bank.
    if len(technical_questions) < 5:
        for skill in skills:
            question = (
                f"What is {skill}, why is it useful, and where have you used it?"
            )
            if question not in technical_questions:
                technical_questions.append(question)
            if len(technical_questions) == 5:
                break

    # Final fallback if no skills were detected.
    fallback_technical = [
        "What are the main data structures you have studied?",
        "How do you analyze the time complexity of an algorithm?",
        "Explain the difference between a process and a thread.",
        "What is DBMS normalization?",
        "Explain the difference between TCP and UDP."
    ]

    for question in fallback_technical:
        if len(technical_questions) == 5:
            break
        if question not in technical_questions:
            technical_questions.append(question)

    project_text = projects.strip() if projects else ""

    if project_text and project_text.lower() != "not found":
        project_reference = project_text.split("\n")[0].strip()
        if len(project_reference) > 100:
            project_reference = project_reference[:100].rstrip() + "..."
    else:
        project_reference = "the project mentioned in your resume"

    project_questions = [
        f"Explain {project_reference} and the problem it solves.",
        "What was your specific contribution to the project?",
        "Which technologies did you use in the project and why did you choose them?",
        "What was the most difficult problem you faced while developing the project?",
        "If you had more time, what improvement would you make to the project?"
    ]

    if experience and experience.strip().lower() != "not found":
        project_questions[3] = (
            "What was the most challenging technical problem you faced during your project or experience?"
        )

    hr_questions = [
        "Tell me about yourself.",
        "Why do you want to work as a software developer?",
        "What are your strengths and how do they help you in software development?",
        "What is one technical skill you are currently improving?",
        "Why should we hire you as a fresher?"
    ]

    return {
        "technical": technical_questions[:5],
        "project": project_questions[:5],
        "hr": hr_questions[:5]
    }


# =========================================================
# GENERATE INTERVIEW QUESTIONS
# =========================================================

@app.route(
    "/generate-interview",
    methods=["POST"]
)
@login_required
def generate_interview():

    resume_text = request.form.get(
        "resume_text",
        ""
    ).strip()

    skills_text = request.form.get(
        "skills",
        ""
    ).strip()

    projects = request.form.get(
        "projects",
        "Not found"
    ).strip()

    experience = request.form.get(
        "experience",
        "Not found"
    ).strip()

    skills = [
        skill.strip()
        for skill in skills_text.split(",")
        if skill.strip()
    ]

    interview_questions = generate_ai_interview_questions(
        resume_text,
        skills,
        projects,
        experience
    )

    return render_template(
        "interview.html",
        interview_questions=interview_questions,
        skills=skills,
        projects=projects,
        experience=experience
    )



# =========================================================
# USER REGISTRATION, LOGIN AND LOGOUT
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password or not confirm_password:
            flash("Please complete all fields.", "danger")
            return render_template("register.html")

        if len(name) > 100:
            flash("Name must be 100 characters or fewer.", "danger")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "danger")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        if User.query.filter_by(email=email).first():
            flash("An account with this email already exists. Please log in.", "warning")
            return redirect(url_for("login"))

        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            session.clear()
            session["user_id"] = user.id
            session["user_name"] = user.name
            session["user_email"] = user.email
            flash(f"Welcome, {user.name}!", "success")
            return redirect(url_for("home"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    # Dashboard analytics are scoped to the currently logged-in user.
    analytics = {
        "total_analyses": 0,
        "average_ats_score": 0,
        "average_job_match": None,
        "recent_analyses": [],
    }

    user_id = session.get("user_id")
    if user_id is not None:
        user_analyses = ResumeAnalysis.query.filter_by(user_id=user_id)
        analytics["total_analyses"] = user_analyses.count()

        average_ats = db.session.query(
            db.func.avg(ResumeAnalysis.ats_score)
        ).filter_by(user_id=user_id).scalar()
        analytics["average_ats_score"] = round(average_ats or 0)

        average_match = db.session.query(
            db.func.avg(ResumeAnalysis.job_match_percentage)
        ).filter(
            ResumeAnalysis.user_id == user_id,
            ResumeAnalysis.job_match_percentage.isnot(None),
        ).scalar()
        analytics["average_job_match"] = (
            round(average_match) if average_match is not None else None
        )

        analytics["recent_analyses"] = (
            user_analyses.order_by(ResumeAnalysis.created_at.desc())
            .limit(5)
            .all()
        )

    return render_template("index.html", analytics=analytics)


# =========================================================
# RESUME ANALYSIS HISTORY
# =========================================================

@app.route("/history")
@login_required
def history():
    analyses = (ResumeAnalysis.query
                .filter_by(user_id=session["user_id"])
                .order_by(ResumeAnalysis.created_at.desc())
                .all())
    return render_template("history.html", analyses=analyses)


# =========================================================
# UPLOAD + ANALYZE
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
@login_required
def upload():

    if "resume" not in request.files:
        flash("Resume file was not received. Please choose a PDF or DOCX file and try again.", "warning")
        return redirect(url_for("home"))

    file = request.files["resume"]

    if not file or not file.filename or not file.filename.strip():
        flash("Please select a resume file before clicking Analyze.", "warning")
        return redirect(url_for("home"))

    original_filename = file.filename.strip()
    if not allowed_file(original_filename):
        flash("Unsupported file type. Please upload a PDF or DOCX file.", "warning")
        return redirect(url_for("home"))

    # Sanitize the name to prevent unsafe paths and use a unique stored name
    # so two uploads with the same filename do not overwrite one another.
    safe_filename = secure_filename(original_filename)
    if not safe_filename:
        flash("The selected filename is invalid. Rename the file and try again.", "warning")
        return redirect(url_for("home"))


    # -----------------------------------------------------
    # JOB DESCRIPTION
    # -----------------------------------------------------

    job_description = request.form.get(
        "job_description",
        ""
    ).strip()


    # -----------------------------------------------------
    # SAVE RESUME
    # -----------------------------------------------------

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_filename)

    try:
        file.save(file_path)
    except OSError as exc:
        app.logger.exception("Could not save uploaded resume")
        flash("The resume could not be saved. Check that the uploads folder is writable, then try again.", "danger")
        return redirect(url_for("home"))


    # -----------------------------------------------------
    # EXTRACT TEXT
    # -----------------------------------------------------

    extension = safe_filename.rsplit(".", 1)[1].lower()

    if extension == "pdf":

        resume_text = extract_pdf_text(
            file_path
        )

    elif extension == "docx":

        resume_text = extract_docx_text(
            file_path
        )

    else:
        flash("Unsupported file format. Please upload a PDF or DOCX file.", "warning")
        return redirect(url_for("home"))

    if not resume_text or not resume_text.strip():
        flash("The file was uploaded, but no readable text was found. For PDF, use a text-based PDF (not a scanned image) or save it as DOCX and retry.", "warning")
        return redirect(url_for("home"))

    # -----------------------------------------------------
    # CLEAN TEXT
    # -----------------------------------------------------

    cleaned_text = clean_text(
        resume_text
    )

    # Keep the latest extracted resume text per user for the AI tools page.
    # It is stored locally and is not sent anywhere unless local Ollama is enabled.
    latest_text_dir = os.path.join(app.root_path, "instance", "resume_texts")
    os.makedirs(latest_text_dir, exist_ok=True)
    with open(os.path.join(latest_text_dir, f"user_{session['user_id']}.txt"), "w", encoding="utf-8") as latest_file:
        latest_file.write(cleaned_text)


    # -----------------------------------------------------
    # PERSONAL INFORMATION
    # -----------------------------------------------------

    name = extract_name(
        cleaned_text
    )

    email = extract_email(
        cleaned_text
    )

    phone = extract_phone(
        cleaned_text
    )


    # -----------------------------------------------------
    # SKILLS
    # -----------------------------------------------------

    skills = detect_skills(
        cleaned_text
    )


    # -----------------------------------------------------
    # SECTIONS
    # -----------------------------------------------------

    education = extract_section(
        cleaned_text,
        [
            "education",
            "academic qualification"
        ]
    )

    experience = extract_section(
        cleaned_text,
        [
            "experience",
            "work experience",
            "professional experience",
            "internship"
        ]
    )

    projects = extract_section(
        cleaned_text,
        [
            "projects",
            "project"
        ]
    )

    certifications = extract_section(
        cleaned_text,
        [
            "certifications",
            "certification"
        ]
    )


    # -----------------------------------------------------
    # QUALITY
    # -----------------------------------------------------

    quality = analyze_resume_quality(
        name,
        email,
        phone,
        skills,
        education,
        experience,
        projects
    )


    # -----------------------------------------------------
    # ATS
    # -----------------------------------------------------

    ats = calculate_ats_score(
        name,
        email,
        phone,
        skills,
        education,
        experience,
        projects,
        certifications
    )


    # -----------------------------------------------------
    # ATS SCORE IMPROVEMENT
    # -----------------------------------------------------

    ats_improvement = generate_ats_improvement(
        name,
        email,
        phone,
        skills,
        education,
        experience,
        projects,
        certifications,
        cleaned_text,
        ats
    )


    # -----------------------------------------------------
    # STRENGTHS
    # -----------------------------------------------------

    strengths = generate_strengths(
        name,
        email,
        phone,
        skills,
        education,
        experience,
        projects,
        certifications,
        ats
    )


    # -----------------------------------------------------
    # WEAKNESSES
    # -----------------------------------------------------

    weaknesses = generate_weaknesses(
        name,
        email,
        phone,
        skills,
        education,
        experience,
        projects,
        certifications,
        cleaned_text,
        ats
    )


    # -----------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------

    recommendations = generate_recommendations(
        skills,
        education,
        experience,
        projects,
        certifications,
        cleaned_text,
        ats
    )


    # -----------------------------------------------------
    # JOB MATCH
    # -----------------------------------------------------

    jd_skills = detect_jd_skills(
        job_description
    )

    (
        job_match_percentage,
        matched_skills,
        missing_skills
    ) = calculate_job_match(
        skills,
        jd_skills
    )


    # -----------------------------------------------------
    # MISSING SKILL RECOMMENDATIONS
    # -----------------------------------------------------

    missing_skill_recommendations = (
        generate_missing_skill_recommendations(
            missing_skills
        )
    )


    # -----------------------------------------------------
    # JOB-SPECIFIC IMPROVEMENTS
    # -----------------------------------------------------

    job_improvement_suggestions = generate_job_improvement_suggestions(
        job_description,
        skills,
        matched_skills,
        missing_skills,
        projects,
        experience
    )


    # -----------------------------------------------------
    # RESUME ANALYTICS DASHBOARD
    # -----------------------------------------------------

    resume_dashboard = generate_resume_dashboard(
        ats,
        job_description,
        job_match_percentage,
        skills,
        education,
        experience,
        projects,
        certifications
    )


    # -----------------------------------------------------
    # AI SUMMARY
    # -----------------------------------------------------

    ai_summary = generate_ai_resume_summary(
        cleaned_text,
        name,
        skills,
        education,
        experience,
        projects
    )


    # -----------------------------------------------------
    # INTERVIEW QUESTIONS
    # -----------------------------------------------------

    interview_questions = {
        "technical": [],
        "project": [],
        "hr": []
    }


    # -----------------------------------------------------
    # SAVE THIS ANALYSIS TO THE LOGGED-IN USER'S HISTORY
    # -----------------------------------------------------

    history_record = ResumeAnalysis(
        user_id=session["user_id"],
        filename=os.path.basename(safe_filename),
        candidate_name=name or "Not detected",
        ats_score=int(ats),
        job_match_percentage=(int(job_match_percentage) if job_description else None),
        skills_json=json.dumps(skills, ensure_ascii=False),
    )
    db.session.add(history_record)
    db.session.commit()

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    return render_template(

        "result.html",

        resume_text=cleaned_text,

        name=name,
        email=email,
        phone=phone,

        skills=skills,

        education=education,
        experience=experience,
        projects=projects,
        certifications=certifications,

        quality=quality,

        ats_score=ats,

        ats_improvement=ats_improvement,

        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recommendations,

        job_description=job_description,

        jd_skills=jd_skills,

        matched_skills=matched_skills,

        missing_skills=missing_skills,

        job_match_percentage=job_match_percentage,

        missing_skill_recommendations=(
            missing_skill_recommendations
        ),

        job_improvement_suggestions=job_improvement_suggestions,

        resume_dashboard=resume_dashboard,

        ai_summary=ai_summary,

        interview_questions=interview_questions
    )


# =========================================================
# HYBRID AI TOOLS (Python NLP + optional local Ollama)
# =========================================================

@app.route("/ai-tools", methods=["GET", "POST"])
@login_required
def ai_tools():
    user_text_path = os.path.join(app.root_path, "instance", "resume_texts", f"user_{session['user_id']}.txt")
    resume_text = ""
    if os.path.isfile(user_text_path):
        try:
            with open(user_text_path, "r", encoding="utf-8") as latest_file:
                resume_text = latest_file.read()
        except OSError:
            app.logger.exception("Could not read latest resume text")
    job_description = ""
    result = None
    action = "analysis"
    if request.method == "POST":
        resume_text = request.form.get("resume_text", resume_text).strip()
        job_description = request.form.get("job_description", "").strip()
        action = request.form.get("action", "analysis")
        if not resume_text:
            flash("Pehle login karke resume upload/analyze karein, ya resume text paste karein.", "warning")
        else:
            if action == "matching":
                if not job_description:
                    flash("Job matching ke liye Job Description paste karein.", "warning")
                else:
                    result = ai_match_job(resume_text, job_description)
            elif action == "improver":
                result = ai_improve_resume(resume_text)
            elif action == "interview":
                result = ai_interview_prep(resume_text, job_description)
            elif action == "roadmap":
                result = ai_learning_roadmap(resume_text, job_description)
            elif action == "dashboard":
                result = ai_analyze_resume(resume_text)
                recent = (ResumeAnalysis.query.filter_by(user_id=session["user_id"])
                          .order_by(ResumeAnalysis.created_at.desc()).limit(10).all())
                result["history"] = [{"date": x.created_at.strftime("%d %b %Y"), "filename": x.filename,
                                       "ats_score": x.ats_score, "job_match": x.job_match_percentage}
                                      for x in recent]
            else:
                result = ai_analyze_resume(resume_text)
    # Provide a personal dashboard snapshot for the dashboard feature.
    history_rows = (ResumeAnalysis.query.filter_by(user_id=session["user_id"])
                    .order_by(ResumeAnalysis.created_at.desc()).limit(10).all())
    return render_template("ai_tools.html", resume_text=resume_text, job_description=job_description,
                           action=action, result=result, history_rows=history_rows,
                           ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))


# =========================================================
# DOWNLOAD RESUME ANALYSIS PDF REPORT
# =========================================================

@app.route("/download-report", methods=["POST"])
@login_required
def download_report():
    """Create a PDF report from the current resume analysis results."""

    def field(name, default="Not provided"):
        value = request.form.get(name, default)
        return value.strip() if value and value.strip() else default

    def list_field(name):
        return [item.strip() for item in request.form.get(name, "").split("||") if item.strip()]

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
        title="Resume Analysis Report"
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"), spaceAfter=18
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", parent=styles["Heading2"],
        textColor=colors.HexColor("#17365D"), spaceBefore=12, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="SmallBody", parent=styles["BodyText"], leading=14, spaceAfter=4
    ))

    story = [Paragraph("Resume Analysis Report", styles["ReportTitle"])]
    story.append(Paragraph("Generated by the Resume Analyzer", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Candidate Information", styles["SectionHeading"]))
    candidate_rows = [
        ["Name", field("candidate_name")],
        ["Email", field("candidate_email")],
        ["Phone", field("candidate_phone")],
    ]
    table = Table(candidate_rows, colWidths=[105, 365], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C7D0D9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(table)

    story.append(Paragraph("2. Scores", styles["SectionHeading"]))
    story.append(Paragraph(f"ATS Score: <b>{escape(field('ats_score', '0'))}/100</b>", styles["SmallBody"]))
    jd_match = field("job_match", "N/A")
    story.append(Paragraph(f"Job Match: <b>{escape(jd_match)}</b>", styles["SmallBody"]))
    story.append(Paragraph(f"Detected Skills: <b>{escape(field('skills_count', '0'))}</b>", styles["SmallBody"]))

    def add_list_section(title, values, empty_text="No items provided."):
        story.append(Paragraph(escape(title), styles["SectionHeading"]))
        if values:
            for value in values:
                story.append(Paragraph("• " + escape(value), styles["SmallBody"]))
        else:
            story.append(Paragraph(escape(empty_text), styles["SmallBody"]))

    add_list_section("3. Skills Detected", list_field("skills"))
    add_list_section("4. Matched Job Skills", list_field("matched_skills"))
    add_list_section("5. Missing Job Skills", list_field("missing_skills"))
    add_list_section("6. Resume Improvement Recommendations", list_field("improvements"))

    story.append(Paragraph("7. Resume Sections", styles["SectionHeading"]))
    for section in ["Education", "Experience", "Projects", "Certifications"]:
        story.append(Paragraph(f"{escape(section)}: {escape(field('section_' + section.lower()))}", styles["SmallBody"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Note: This report is based on automated text and keyword checks. Review the results manually before using them for job applications.",
        styles["Italic"]
    ))

    document.build(story)
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="resume_analysis_report.pdf",
        mimetype="application/pdf"
    )


# =========================================================
# RUN APPLICATION
# =========================================================

@app.errorhandler(413)
def uploaded_file_too_large(_error):
    flash("The selected file is too large. Please upload a PDF or DOCX smaller than 10 MB.", "warning")
    return redirect(url_for("home"))


if __name__ == "__main__":

    app.run(
        debug=True
    )