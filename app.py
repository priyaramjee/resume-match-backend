from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
import json
import re

from google import genai
from google.genai.types import GenerateContentConfig

# -------------------------------------------------
# App setup
# -------------------------------------------------

app = FastAPI(title="Resume Match API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Request model
# -------------------------------------------------

class AnalyzeRequest(BaseModel):
    resume_text: str
    job_text: str

# -------------------------------------------------
# Health check
# -------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------
# Thread pool
# -------------------------------------------------

executor = ThreadPoolExecutor(max_workers=2)

# -------------------------------------------------
# JSON extraction helper
# -------------------------------------------------

def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Gemini output")
    return json.loads(match.group())

# -------------------------------------------------
# Gemini runner (SAFE)
# -------------------------------------------------

def run_gemini(resume_text: str, job_text: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an expert resume reviewer.

Compare the resume and job description below.

Return ONLY a valid JSON object.
Rules:
- match_score: number between 0 and 100
- missing_skills: max 5 items
- key_strengths: max 5 items
- improvement_suggestions: max 5 items
- Use short phrases
- Do NOT add extra keys
- Do NOT add explanations
- Do NOT use markdown

Resume:
{resume_text}

Job Description:
{job_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0.2,
            max_outpu_
