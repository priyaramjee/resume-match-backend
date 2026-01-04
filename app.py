from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional, Dict, Any, List
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
    return {"version": "skills-v1-explainable-score"}

executor = ThreadPoolExecutor(max_workers=2)

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _call_gemini(client, prompt: str, schema: Optional[Schema], max_tokens: int) -> str:
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

def _safe_json_load(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    raw = raw.strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def _dedupe_str_list(seq: Any) -> List[str]:
    if not isinstance(seq, list):
        return []
    out: List[str] = []
    seen = set()
    for x in seq:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

# -------------------------------------------------
# Gemini runner: SKILLS + EVIDENCE (no match_score from Gemini)
# -------------------------------------------------

def run_gemini(resume_text: str, job_text: str) -> Dict[str, Any]:
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
                        "category": Schema(type="string"),
                        "status": Schema(type="string"),
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
        "- evidence <= 12 words\n"
        "- short paraphrase from resume/job\n\n"
        "Rules:\n"
        "- Use Title Case skill names\n"
        "- Do NOT invent skills not in the job description\n"
        "- improvement_suggestions: max 5\n\n"
        f"RESUME:\n{resume_text}\n\n"
        f"JOB DESCRIPTION:\n{job_text}\n"
    )

    # Attempt 1
    raw1 = _call_gemini(client, base_prompt, skills_schema, 900)
    print("RAW GEMINI OUTPUT (1):\n", raw1)
    obj1 = _safe_json_load(raw1)
    if obj1 and "must_have_skills" in obj1:
        return obj1

    # Attempt 2 stronger
    prompt2 = base_prompt + "\nIMPORTANT: Output must be COMPLETE valid JSON. No extra text.\n"
    raw2 = _call_gemini(client, prompt2, skills_schema, 900)
    print("RAW GEMINI OUTPUT (2):\n", raw2)
    obj2 = _safe_json_load(raw2)
    if obj2 and "must_have_skills" in obj2:
        return obj2

    # Attempt 3 minimal
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
        "Return only keys: must_have_skills, nice_to_have_skills, matched_skills, missing_skills\n\n"
        f"RESUME:\n{resume_text}\n\n"
        f"JOB DESCRIPTION:\n{job_text}\n"
    )
    raw3 = _call_gemini(client, prompt3, minimal_schema, 350)
    print("RAW GEMINI OUTPUT (3):\n", raw3)
    obj3 = _safe_json_load(raw3)

    if obj3 and "must_have_skills" in obj3:
        must = obj3.get("must_have_skills", []) or []
        nice = obj3.get("nice_to_have_skills", []) or []
        matched = {s.lower() for s in _dedupe_str_list(obj3.get("matched_skills", []))}

        evidence = []
        for s in _dedupe_str_list(must):
            evidence.append({
                "skill": s,
                "category": "must_have",
                "status": "matched" if s.lower() in matched else "missing",
                "evidence": "Derived from resume/job comparison",
            })
        for s in _dedupe_str_list(nice):
            evidence.append({
                "skill": s,
                "category": "nice_to_have",
                "status": "matched" if s.lower() in matched else "missing",
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

    # Final fallback
    return {
        "must_have_skills": [],
        "nice_to_have_skills": [],
        "matched_skills": [],
        "missing_skills": [],
        "skill_evidence": [],
        "improvement_suggestions": ["Temporary model issue. Please try again."],
    }

# -------------------------------------------------
# Analyze endpoint: EXPLAINABLE SCORE
# -------------------------------------------------

@app.post("/analyze")
def analyze_resume(payload: AnalyzeRequest):
    future = executor.submit(run_gemini, payload.resume_text, payload.job_text)

    try:
        result = future.result(timeout=25) or {}

        must = _dedupe_str_list(result.get("must_have_skills", []))
        nice = _dedupe_str_list(result.get("nice_to_have_skills", []))
        matched = _dedupe_str_list(result.get("matched_skills", []))
        missing = _dedupe_str_list(result.get("missing_skills", []))
        breakdown = result.get("skill_evidence", []) or []
        suggestions = _dedupe_str_list(result.get("improvement_suggestions", []))

        matched_set = {s.lower() for s in matched}

        must_total = len(must)
        nice_total = len(nice)
        must_matched = sum(1 for s in must if s.lower() in matched_set)
        nice_matched = sum(1 for s in nice if s.lower() in matched_set)

        weights = {"must_have": 0.8, "nice_to_have": 0.2}
        must_ratio = (must_matched / must_total) if must_total else 0.0
        nice_ratio = (nice_matched / nice_total) if nice_total else 0.0

        score = 100.0 * (weights["must_have"] * must_ratio + weights["nice_to_have"] * nice_ratio)
        score = int(round(min(max(score, 0.0), 100.0)))

        out = {
            "match_score": score,
            "matched_skills": matched,
            "missing_skills": missing,
            "skill_breakdown": breakdown,
            "score_explanation": {
                "must_have_total": must_total,
                "must_have_matched": must_matched,
                "nice_to_have_total": nice_total,
                "nice_to_have_matched": nice_matched,
                "weights": weights,
                "calculation": (
                    f"Score = 100 * (0.8*({must_matched}/{must_total or 1}) "
                    f"+ 0.2*({nice_matched}/{nice_total or 1}))"
                ),
            },
            "improvement_suggestions": suggestions[:5],
        }

        print("RETURNING TO CLIENT:", out)
        return out

    except TimeoutError:
        out = {
            "match_score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "skill_breakdown": [],
            "score_explanation": {"error": "Timed out. Please retry."},
            "improvement_suggestions": ["Timed out. Please retry."],
        }
        print("RETURNING TO CLIENT (TIMEOUT):", out)
        return out

    except Exception as e:
        out = {
            "match_score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "skill_breakdown": [],
            "score_explanation": {"error": f"Error: {str(e)}"},
            "improvement_suggestions": ["Temporary error. Please retry."],
        }
        print("RETURNING TO CLIENT (ERROR):", out)
        return out
