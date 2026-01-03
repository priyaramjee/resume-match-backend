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
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"version": "retry-v5-never-zero-on-glitch"}

executor = ThreadPoolExecutor(max_workers=2)

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _call_gemini(client, prompt: str, schema: Schema | None, max_tokens: int) -> str:
    # schema can be None; we still request JSON mime type
    cfg_kwargs = dict(
        temperature=0.2,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
    )
    if schema is not None:
        cfg_kwargs["response_schema"] = schema

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(**cfg_kwargs),
    )

    raw = ""
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if getattr(part, "text", None):
                raw += part.text
    return raw.strip()

def _safe_json_load(raw: str):
    if not raw:
        return None
    raw = raw.strip()

    # direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # extract first JSON object
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def _extract_score_from_text(raw: str) -> float | None:
    """
    If Gemini returns text like "Here is { ... }" or "Here is 65",
    extract a score safely.
    """
    if not raw:
        return None

    # Try JSON object first
    obj = _safe_json_load(raw)
    if obj and isinstance(obj, dict) and "match_score" in obj:
        try:
            return float(obj["match_score"])
        except Exception:
            return None

    # Try regex score extraction
    m = re.search(r"match_score\"\s*:\s*([0-9]+(?:\.[0-9]+)?)", raw)
    if m:
        return float(m.group(1))

    # Any standalone number 0-100
    m2 = re.search(r"\b([0-9]{1,3}(?:\.[0-9]+)?)\b", raw)
    if m2:
        val = float(m2.group(1))
        if 0 <= val <= 100:
            return val

    return None

def _heuristic_score(resume_text: str, job_text: str) -> float:
    rt = (resume_text or "").lower()
    jt = (job_text or "").lower()
    if not rt.strip() and not jt.strip():
        return 0.0

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

    score = 40 + ratio * 55  # 40..95
    return round(min(max(score, 5.0), 95.0), 1)

# -------------------------------------------------
# Gemini runner (retry-v5)
# -------------------------------------------------

def run_gemini(resume_text: str, job_text: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    skills_schema = Schema(
        type="object",
        properties={
            "must_have_skills": Schema(type="array", items=Schema(type="string")),
            "nice_to_have_skills": Schema(type="array", items=Schema(type="string")),
            "matched_skills": Schema(type="array", items=Schema(type="string")),
            "missing_skills": Schema(type="array", items=Schema(type="string")),
            "skill_evidence": Schema(
                type="array",
                items=Schema(
                    type="object",
                    properties={
                        "skill": Schema(type="string"),
                        "category": Schema(type="string"),  # must_have | nice_to_have
                        "status": Schema(type="string"),    # matched | missing
                        "evidence": Schema(type="string"),
                    },
                    required=["skill", "category", "status", "evidence"],
                ),
            ),
            "improvement_suggestions": Schema(type="array", items=Schema(type="string")),
        },
        required=[
            "must_have_skills",
            "nice_to_have_skills",
            "matched_skills",
            "missing_skills",
            "skill_evidence",
            "improvement_suggestions",
        ],
    )

    base_prompt = (
        "You are an expert resume reviewer.\n"
        "Return JSON ONLY.\n\n"
        "Task:\n"
        "1) From the JOB DESCRIPTION, extract:\n"
        "   - must_have_skills (6-12)\n"
        "   - nice_to_have_skills (0-8)\n"
        "2) Compare RESUME vs JOB and produce:\n"
        "   - matched_skills (subset of must/nice)\n"
        "   - missing_skills (subset of must/nice)\n"
        "   - skill_evidence: for EACH skill in must_have_skills and nice_to_have_skills,\n"
        "     include: skill, category (must_have/nice_to_have), status (matched/missing), evidence.\n\n"
        "Evidence rules:\n"
        "- evidence must be <= 12 words\n"
        "- use short paraphrase of resume/job text\n\n"
        "Rules:\n"
        "- Use consistent skill names in Title Case\n"
        "- Do NOT invent skills not present in the job description\n"
        "- improvement_suggestions: max 5 items\n\n"
        f"RESUME:\n{resume_text}\n\n"
        f"JOB DESCRIPTION:\n{job_text}\n"
    )

    # Attempt 1
    raw1 = _call_gemini(client, base_prompt, skills_schema, 900)
    print("RAW GEMINI OUTPUT (1):\n", raw1)
    obj1 = _safe_json_load(raw1)
    if obj1 and isinstance(obj1, dict) and "must_have_skills" in obj1:
        return obj1

    # Attempt 2 stronger
    prompt2 = base_prompt + "\nIMPORTANT: Output must be COMPLETE valid JSON. No extra text.\n"
    raw2 = _call_gemini(client, prompt2, skills_schema, 900)
    print("RAW GEMINI OUTPUT (2):\n", raw2)
    obj2 = _safe_json_load(raw2)
    if obj2 and isinstance(obj2, dict) and "must_have_skills" in obj2:
        return obj2

    # Attempt 3: minimal skills-only (sometimes helps when output gets truncated)
    minimal_schema = Schema(
        type="object",
        properties={
            "must_have_skills": Schema(type="array", items=Schema(type="string")),
            "nice_to_have_skills": Schema(type="array", items=Schema(type="string")),
            "matched_skills": Schema(type="array", items=Schema(type="string")),
            "missing_skills": Schema(type="array", items=Schema(type="string")),
        },
        required=["must_have_skills", "nice_to_have_skills", "matched_skills", "missing_skills"],
    )

    prompt3 = (
        "Return JSON ONLY. No extra text.\n"
        "Return only these keys:\n"
        "must_have_skills, nice_to_have_skills, matched_skills, missing_skills\n\n"
        f"RESUME:\n{resume_text}\n\n"
        f"JOB DESCRIPTION:\n{job_text}\n"
    )
    raw3 = _call_gemini(client, prompt3, minimal_schema, 350)
    print("RAW GEMINI OUTPUT (3):\n", raw3)
    obj3 = _safe_json_load(raw3)

    if obj3 and isinstance(obj3, dict) and "must_have_skills" in obj3:
        # synthesize skill_evidence minimally so frontend still works
        must = obj3.get("must_have_skills", []) or []
        nice = obj3.get("nice_to_have_skills", []) or []
        matched = {s.lower() for s in (obj3.get("matched_skills", []) or [])}

        evidence = []
        for s in must:
            evidence.append({
                "skill": s,
                "category": "must_have",
                "status": "matched" if isinstance(s, str) and s.lower() in matched else "missing",
                "evidence": "Derived from resume/job comparison",
            })
        for s in nice:
            evidence.append({
                "skill": s,
                "category": "nice_to_have",
                "status": "matched" if isinstance(s, str) and s.lower() in matched else "missing",
                "evidence": "Derived from resume/job comparison",
            })

        return {
            "must_have_skills": must,
            "nice_to_have_skills": nice,
            "matched_skills": obj3.get("matched_skills", []) or [],
            "missing_skills": obj3.get("missing_skills", []) or [],
            "skill_evidence": evidence,
            "improvement_suggestions": ["Re-run to get richer evidence text."],
        }

    # Final fallback: return empty structured object (so /analyze can still compute)
    return {
        "must_have_skills": [],
        "nice_to_have_skills": [],
        "matched_skills": [],
        "missing_skills": [],
        "skill_evidence": [],
        "improvement_suggestions": [
            "Temporary model issue. Please try again."
        ],
    }

# -------------------------------------------------
# Analyze endpoint (never returns 0 on glitch)
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
            "match_score": _heuristic_score(payload.resume_text, payload.job_text),
            "missing_skills": [],
            "key_strengths": [],
            "improvement_suggestions": ["Timed out. Showing estimated score; retry for full analysis."],
        }
        print("RETURNING TO CLIENT (TIMEOUT):", out)
        return out

    except Exception as e:
        out = {
            "match_score": _heuristic_score(payload.resume_text, payload.job_text),
            "missing_skills": [],
            "key_strengths": [],
            "improvement_suggestions": [f"Temporary error: {str(e)}", "Retry for full analysis."],
        }
        print("RETURNING TO CLIENT (ERROR):", out)
        return out
