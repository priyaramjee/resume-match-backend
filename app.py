from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
import json

from google import genai
from google.genai.types import GenerateContentConfig, Schema

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

@app.get("/version")
def version():
    return {"version": "retry-v3-with-fallback"}

executor = ThreadPoolExecutor(max_workers=2)

def _call_gemini(client, prompt: str, schema: Schema, max_tokens: int = 500) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    raw = ""
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if getattr(part, "text", None):
                raw += part.text
    return raw.strip()

def run_gemini(resume_text: str, job_text: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    full_schema = Schema(
        type="object",
        properties={
            "match_score": Schema(type="number"),
            "missing_skills": Schema(type="array", items=Schema(type="string")),
            "key_strengths": Schema(type="array", items=Schema(type="string")),
            "improvement_suggestions": Schema(type="array", items=Schema(type="string")),
        },
        required=["match_score", "missing_skills", "key_strengths", "improvement_suggestions"],
    )

    score_only_schema = Schema(
        type="object",
        properties={"match_score": Schema(type="number")},
        required=["match_score"],
    )

    base_prompt = (
        "You are an expert resume reviewer.\n"
        "Return JSON ONLY.\n"
        "Limits:\n"
        "- missing_skills: max 5 items\n"
        "- key_strengths: max 4 items\n"
        "- improvement_suggestions: max 4 items\n"
        "- Each item max 8 words\n\n"
        f"Resume:\n{resume_text}\n\n"
        f"Job Description:\n{job_text}\n"
    )

    # Attempt 1
    raw1 = _call_gemini(client, base_prompt, full_schema, max_tokens=450)
    print("RAW GEMINI OUTPUT (1):\n", raw1)
    try:
        return json.loads(raw1)
    except Exception:
        pass

    # Attempt 2 (stronger)
    prompt2 = base_prompt + "\nIMPORTANT: Output must be COMPLETE valid JSON. Do not stop early.\n"
    raw2 = _call_gemini(client, prompt2, full_schema, max_tokens=450)
    print("RAW GEMINI OUTPUT (2):\n", raw2)
    try:
        return json.loads(raw2)
    except Exception:
        pass

    # Attempt 3 (fallback: score only)
    prompt3 = (
        "Return JSON ONLY with exactly one key: match_score (0-100). "
        "No other keys.\n\n"
        f"Resume:\n{resume_text}\n\n"
        f"Job Description:\n{job_text}\n"
    )
    raw3 = _call_gemini(client, prompt3, score_only_schema, max_tokens=60)
    print("RAW GEMINI OUTPUT (3):\n", raw3)

    score_obj = json.loads(raw3)
    # Fill missing arrays so the frontend never breaks
    return {
        "match_score": score_obj.get("match_score", 0),
        "missing_skills": [],
        "key_strengths": [],
        "improvement_suggestions": [
            "Partial response: try again for full details."
        ],
    }

@app.post("/analyze")
def analyze_resume(payload: AnalyzeRequest):
    future = executor.submit(run_gemini, payload.resume_text, payload.job_text)

    try:
        result = future.result(timeout=20)

        out = {
            "match_score": float(result.get("match_score", 0)),
            "missing_skills": result.get("missing_skills", []) or [],
            "key_strengths": result.get("key_strengths", []) or [],
            "improvement_suggestions": result.get("improvement_suggestions", []) or [],
        }

        print("RETURNING TO CLIENT:", out)
        return out

    except TimeoutError:
        out = {
            "match_score": 0,
            "missing_skills": [],
            "key_strengths": [],
            "improvement_suggestions": ["Analysis timed out. Please try again."],
        }
        print("RETURNING TO CLIENT (TIMEOUT):", out)
        return out

    except Exception as e:
        out = {
            "match_score": 0,
            "missing_skills": [],
            "key_strengths": [],
            "improvement_suggestions": [f"Analysis failed: {str(e)}"],
        }
        print("RETURNING TO CLIENT (ERROR):", out)
        return out
