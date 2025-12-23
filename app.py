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
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Gemini output")
    return json.loads(match.group())

# -------------------------------------------------
# Gemini runner (SAFE + COMPLETE)
# -------------------------------------------------

def run_gemini(resume_text: str, job_text: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    prompt = (
        "You are an expert resume reviewer.\n\n"
        "Compare the resume and job description below.\n\n"
        "Return ONLY a valid JSON object.\n"
        "Rules:\n"
        "- match_score: number between 0 and 100\n"
        "- missing_skills: max 5 items\n"
        "- key_strengths: max 5 items\n"
        "- improvement_suggestions: max 5 items\n"
        "- Use short phrases\n"
        "- Do NOT add extra keys\n"
        "- Do NOT add explanations\n"
        "- Do NOT use markdown\n\n"
        f"Resume:\n{resume_text}\n\n"
        f"Job Description:\n{job_text}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1000
        )
    )

    # IMPORTANT: assemble text from candidates.parts
    raw = ""
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                raw += part.text

    print("RAW GEMINI OUTPUT:\n", raw)

    # Guard against truncation
    if raw.count("{") != raw.count("}"):
        raise RuntimeError("Incomplete JSON returned by Gemini")

    return extract_json(raw)

# -------------------------------------------------
# Analyze endpoint (GUARANTEED RESPONSE)
# -------------------------------------------------

@app.post("/analyze")
def analyze_resume(payload: AnalyzeRequest):
    future = executor.submit(
        run_gemini,
        payload.resume_text,
        payload.job_text,
    )

    try:
        result = future.result(timeout=15)

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
