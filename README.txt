ResumeIQ Hybrid AI Integration Update

This package is based on your saved app(20261019-171049).py and preserves its existing routes/features. It adds: /ai-tools, ai_features.py, and templates/ai_tools.html.

INSTALL / MERGE
1. Back up your current project folder.
2. Copy the included app.py over your current app.py (or merge the /ai-tools route and ai_features imports if your current file is newer).
3. Copy ai_features.py beside app.py.
4. Copy templates/ai_tools.html into your existing templates folder. Do not delete/replace your other templates, static, database, uploads, .env, or SQLite database.
5. Start the app using your existing venv and `python app.py`. Log in and open `/ai-tools`.

FEATURES
- Smart analysis: uses existing PDF/DOCX extraction, name/skills/sections and ATS scoring.
- Job matching: uses existing skills detector and job match calculation, plus optional local model explanation.
- Resume improver: optional local model or safe template fallback.
- Mock interview: optional local model or built-in questions.
- Learning roadmap: skill-gap-based plan.
- Dashboard: reads analysis history for the signed-in user.

OPTIONAL FREE LOCAL GENERATIVE MODEL
- The AI features do not require a paid API key. Without a model, rule-based/local fallback works.
- For richer generation, install Ollama from https://ollama.com, then run `ollama pull llama3.2:3b`. Start Ollama and restart Flask.
- Optional .env settings: `OLLAMA_MODEL=llama3.2:3b` and `OLLAMA_URL=http://127.0.0.1:11434/api/generate`.
- Resume text is sent only to the local Ollama endpoint on your machine. If Ollama is not running, fallback text is used.

NOTES
- The existing dashboard/history page remains in place; the new AI tools page is separate to minimize disruption. Use `/ai-tools` after login.
- PDF text extraction does not OCR image-only/scanned PDFs.
- ATS score and skill match are heuristics, not a hiring prediction. Check generated text for accuracy before using it.
