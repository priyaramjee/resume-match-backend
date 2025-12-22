from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import google.generativeai as genai

app = FastAPI()

# CORS (required for Lovable)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Models ---------

class AnalyzeRequest(BaseModel):
    resume_text: str
    job_text: str


# --------- Health check (VERY important) ---------

@app.get("/health")
def health():
    return {"status": "ok"}


# --------- Analyze endpoint ---------

@app.post("/analyze")
def analyze_resume(payload: AnalyzeRequest):

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {
                "error": "Missing GEMINI_API_KEY environment variable"
            }

        # Initialize Gemini INSIDE the request (never at import time)
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
You are an expert resume reviewer.

Compare the resume and job description below and return STRICT JSON only
with these keys:
- match_score (number 0-100)
- missing_skills (array of strings)
- key_strengths (array of strings)
- improvement_suggestions (array of strings)

Resume:
{payload.resume_text}

Job Description:
{payload.job_text}
"""

        response = model.generate_content(
            prompt,
            request_options={"timeout": 15}
        )

        text = response.text.strip()

        return text

    except Exception as e:
        return {
            "error": "Analysis failed",
            "details": str(e)
        }
