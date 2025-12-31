from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
import json
import re

from google import genai
from google.genai.types import GenerateContentConfig, Schema

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

class AnalyzeRequest(BaseModel):
    resume_text: str
    job_text: str

@app.get("/")
def root():
    return {"status": "ok", "service": "resume-match-backend"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"version": "retry-v4-json-safe-fallback"}

executor = ThreadPoolExecutor(max_workers=2)

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _call_gemini(client, prompt: str, schema: Schema, max_tokens: int) -> str:
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

def _safe_json_load(raw: str):
    """
    Try to parse JSON. Also tries to extract the first {...} object if there is extra text.
    Returns dict or None.
    """
    if not raw:
        return None
    raw = raw.strip()

    # direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # extract first json object
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    return None

def _heuristic_score(resume_text: str, job_text: str) -> float:
    """
    Deterministic fallback score (never returns 0 unless both inputs are empty).
    Simple keyword overlap ratio with guardrails.
    """
    rt = (resume_text or "").lower()
    jt = (job_text or "").lower()
    if not rt.strip() and not jt.strip():
        return 0.0

    # Common keywords for your use case (adjust anytime)
    keywords = [
        "sql", "databricks", "python", "cloud", "analytics", "manager", "leadership",
        "stakeholder", "dashboard", "finance", "operations", "spark", "azure", "team"
    ]
    in_resume = {k for k in keywords if k in rt}
    in_job = {k for k in keywords if k in jt}
    if not in_job:
        return 50.0

    overlap = len(in_resume.intersection(in_job))
    ratio = overlap / max(len(in_job), 1)

    # Map overlap ratio to a reasonable range
    score = 40 + ratio * 55  # => 40..95
    return round(min(max(score, 5.0), 95.0), 1)

# -------------------------------------------------
# Gemini runner (RETRY-V4)
# -------------------------------------------------

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
    raw1 = _call_gemini(client, base_prompt, full_schema, 450)
    print("RAW GEMINI OUTPUT (1):\n", raw1)
    obj1 = _safe_json_load(raw1)
    if obj1 and "match_score" in obj1:
        return obj1

    # Attempt 2 (stronger)
    prompt2 = base_prompt + "\nIMPORTANT: Output must be COMPLETE valid JSON. Do not stop early.\n"
    raw2 = _call_gemini(client, prompt2, full_schema, 450)
    print("RAW GEMINI OUTPUT (2):\n", raw2)
    obj2 = _safe_json_load(raw2)
    if obj2 and "match_score" in obj2:
        return obj2

    # Attempt 3 (score-only strict)
    prompt3 = (
        "RETURN ONLY JSON. No words.\n"
        "Output exactly like this example: {\"match_score\": 72}\n"
        "No extra keys.\n\n"
        f"Resume:\n{resume_text}\n\n"
        f"Job Description:\n{job_text}\n"
    )
    raw3 = _call_gemini(client, prompt3, score_only_schema, 80)
    print("RAW GEMINI OUTPUT (3):\n", raw3)
    obj3 = _safe_json_load(raw3)

    if obj3 and "match_score" in obj3:
        return {
            "match_score": obj3.get("match_score", 0),
            "missing_skills": [],
            "key_strengths": [],
            "improvement_suggestions": ["Partial response returned. Re-run for full analysis."],
        }

    # Final deterministic fallback (never break UI)
    score = _heuristic_score(resume_text, job_text)
    return {
        "match_score": score,
        "missing_skills": [],
        "key_strengths": [],
        "improvement_suggestions": [
            "Temporary model issue. Score estimated from keyword overlap.",
            "Re-run for full analysis."
        ],
    }

# -------------------------------------------------
# Analyze endpoint
# -------------------------------------------------

@app.post("/analyze")
def analyze_resume(payload: AnalyzeRequest):
    future = executor.submit(run_gemini, payload.resume_text, payload.job_text)

    try:
        result = future.result(timeout=25)

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
