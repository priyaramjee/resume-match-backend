from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from google import genai
from google.genai.types import GenerateContentConfig

# --------------------
# FastAPI app
# --------------------

app = FastAPI(title="Resume Match API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # OK for MVP
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Request model
# --------------------

class AnalyzeRequest(BaseModel):
    resume_text: str
    job_text: str


# --------------------
# Health check
# --------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------
# Initialize Gemini client ONCE
# (safe with google.genai)
# --------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # Fail fast if misconfigured
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)


# --------------------
# Analyze endpoint
# --------------------

@app.post("/analyze")
def analyze_resume(payload: AnalyzeRequest):
    return {
        "match_score": 92,
        "missing_skills": ["Python"],
        "key_strengths": [
            "Strong SQL experience",
            "Databricks and cloud analytics background",
            "Senior stakeholder management"
        ],
        "improvement_suggestions": [
            "Add Python experience if applicable",
            "Quantify business impact in recent roles"
        ]
    }

You are an expert resume reviewer.

Compare the resume and the job description below.

Return STRICT JSON only with
