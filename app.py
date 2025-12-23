from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
import json
import re

from google import genai
from google.genai.types import GenerateContentConfig

app = FastAPI(title="Resume Match API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    resume_text: str
    job_text: str

@app.get("/health")
def health():
    return {"status": "ok"}

executor = ThreadPoolExecutor(max_workers=2)

def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in Gemini output")
    return json.loads(match.group())

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
- Do NOT add extra keys or explanations

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
            max_output_tokens=1000,
        ),
    )

    raw = response.text
    print("RAW GEMINI OUTPUT:\n", raw)

    if raw.count("{") != raw.count("}"):
        raise RuntimeError("Incomplete JSON returned by Gemini")

    return extract_json(raw)

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
