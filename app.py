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
    allow_origins=["*"],   # OK for MVP
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
# Thread pool (prevents blocking)
# -------------------------------------------------

executor = ThreadPoolExecutor(max_workers=2)

# -------------------------------------------------
# Helper: robust JSON extraction
# -------------------------------------------------

def extract_json(text: str) -> dict:
    """
    Extract the first JSON object found in the text.
    Handles markdown fences and extra commentary.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Gemini output")
    return json.loads(match.group())

# -------------------------------------------------
# Gemini runner (off main thread)
# -------------------------------------------------

def run_gemini(resume_text: str, job_text: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an expert resume reviewer.

Compare the resume and job description below.

Return ONLY a valid JSON object (no markdown, no explanations)
with exactly these keys:
- match_score (number 0-100)
- missing_skills (array of strings)
- key_strengths (array of strings)
- improvement_suggestions (array of strings)

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
            max_output_tokens=600,
        ),
    )

    raw = response.text
    print("RAW GEMINI OUTPUT:\n", raw)

    return extract_json(raw)

# -------------------------------------------------
# Analyze endpoint (SAFE + GUARANTEED RESPONSE)
# -------------------------------------------------

@app.post("/analyze")
def analyze_resume(payload: AnalyzeRequest):

    future = executor.submit(
        run_gemini,
        payload.resume_text,
        payload.job_text,
    )

    try:
        result = future.result(timeout=12)

        # Always return UI-safe schema
        return {
            "match_score": result.get("match_score", 0),
            "missing_skills": result.get("missing_skills", []),
            "key_strengths": result.get("key_strengths", []),
            "improvement_suggestions": result.get("improvement_suggestions", []),
        }

    except TimeoutError:
        return {
            "match_score": 0,
            "missing_skills": [],
            "key_strengths": [],
            "improvement_suggestions": [
                "Analysis timed out. Please try again."
            ],
        }

    except Exception as e:
        return {
            "match_score": 0,
            "missing_skills": [],
            "key_strengths": [],
            "improvement_suggestions": [
                f"Analysis failed: {str(e)}"
            ],
        }
