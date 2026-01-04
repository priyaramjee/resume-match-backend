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
    return {"version": "skills-v3-stop-seq + repair-fallback + explainable-score"}

executor = ThreadPoolExecutor(max_workers=2)

# -------------------------------------------------
# Helpers
# -------------------------------------------------

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

def _safe_json_load(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None

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

def _strip_code_fences(s: str) -> str:
    if not s or not isinstance(s, str):
        return s
    s = s.strip()
    # remove ```json ... ```
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _try_parse_any(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or not isinstance(raw, str):
        return None
    raw = _strip_code_fences(raw)
    obj = _safe_json_load(raw)
    if obj and isinstance(obj, dict):
        return obj
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return _safe_json_load(m.group(0))
    return None

def _call_gemini(client, prompt: str, schema: Optional[Schema], max_tokens: int) -> str:
    """
    Safe Gemini call:
    - Handles cases where response.candidates is None/empty
    - Uses stop sequences to reduce preambles like "Here is the JSON..."
    - Falls back to response.text if available
    """
    cfg_kwargs = dict(
        temperature=0.0,  # stronger compliance
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        stop_sequences=["\n\nHere is", "Here is", "```"],
    )
    if schema is not None:
        cfg_kwargs["response_schema"] = schema

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(**cfg_kwargs),
    )

    raw_parts: List[str] = []

    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                raw_parts.append(text)

    raw = "".join(raw_parts).strip()

    # fallback to response.text
    if not raw:
        txt = getattr(response, "text", None)
        if isinstance(txt, str) and txt.strip():
            raw = txt.strip()

    if not raw:
        print("DEBUG: Gemini returned empty text. candidates_len=", len(candidates))

    return raw

def _ensure_json_via_repair(client, raw_text: str, target_schema: Schema) -> Optional[Dict[str, Any]]:
    """
    If Gemini returns non-JSON like 'Here is the JSON requested', run a tiny repair prompt
    that outputs strict JSON ONLY (or an empty valid JSON object).
    """
    if not raw_text or not isinstance(raw_text, str):
        raw_text = ""

    repair_prompt = (
        "Return JSON ONLY. No extra text.\n"
        "Convert the following text into valid JSON that matches the required schema.\n"
        "If the text does not contain JSON, output an empty valid JSON object that matches schema defaults.\n\n"
        f"TEXT:\n{raw_text}\n"
    )

    raw = _call_gemini(client, repair_prompt, target_schema, 250)
    raw = (raw or "").strip()

    obj = _try_parse_any(raw)
    if obj and isinstance(obj, dict):
        return obj

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
# Gemini runner: two-step (lists -> classify) + repair fallback
# -------------------------------------------------

def run_gemini(resume_text: str, job_text: str) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    # -------- Step A: extract ONLY skill lists (tiny output) --------
    list_schema = Schema(
        type="object",
        properties={
            "must_have_skills": Schema(type="array", items=Schema(type="string")),
            "nice_to_have_skills": Schema(type="array", items=Schema(type="string")),
        },
        required=["must_have_skills", "nice_to_have_skills"],
    )

    prompt_a = (
        "Return JSON ONLY. No extra text.\n"
        "Extract skills as short nouns (1-3 words). Avoid long phrases.\n"
        "Output exactly two keys: must_have_skills, nice_to_have_skills.\n"
        "Limits: must_have_skills 5-8 items; nice_to_have_skills 0-4 items.\n\n"
        f"JOB DESCRIPTION:\n{job_text}\n"
    )

    raw_a1 = _call_gemini(client, prompt_a, list_schema, 220)
    print("RAW GEMINI SKILL LIST (A1):\n", raw_a1)
    obj_a = _try_parse_any(raw_a1)

    if not (obj_a and isinstance(obj_a, dict) and "must_have_skills" in obj_a):
        raw_a2 = _call_gemini(client, prompt_a + "\nNO markdown. NO extra text.\n", list_schema, 220)
        print("RAW GEMINI SKILL LIST (A2):\n", raw_a2)
        obj_a = _try_parse_any(raw_a2)

        # Repair fallback
        if not (obj_a and isinstance(obj_a, dict) and "must_have_skills" in obj_a):
            repaired = _ensure_json_via_repair(client, raw_a2 or raw_a1, list_schema)
            if repaired and "must_have_skills" in repaired:
                obj_a = repaired

    if not (obj_a and isinstance(obj_a, dict) and "must_have_skills" in obj_a):
        return {
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "matched_skills": [],
            "missing_skills": [],
            "skill_evidence": [],
            "improvement_suggestions": ["Temporary model issue. Please try again."],
        }

    must = _dedupe_str_list(obj_a.get("must_have_skills", []))
    nice = _dedupe_str_list(obj_a.get("nice_to_have_skills", []))

    # -------- Step B: classify matched vs missing (tiny output) --------
    classify_schema = Schema(
        type="object",
        properties={
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
        required=["matched_skills", "missing_skills", "skill_evidence", "improvement_suggestions"],
    )

    prompt_b = (
        "Return JSON ONLY. No extra text.\n"
        "Using the skill lists from the job, decide which are matched in the resume.\n"
        "Rules:\n"
        "- matched_skills and missing_skills must only use skills from the provided lists.\n"
        "- skill_evidence must include ONLY must_have skills.\n"
        "- For each must-have: {skill, category:\"must_have\", status:\"matched\"|\"missing\", evidence}\n"
        "- evidence <= 10 words\n"
        "- improvement_suggestions: max 4\n\n"
        f"MUST_HAVE_SKILLS:\n{json.dumps(must)}\n\n"
        f"NICE_TO_HAVE_SKILLS:\n{json.dumps(nice)}\n\n"
        f"RESUME:\n{resume_text}\n"
    )

    raw_b1 = _call_gemini(client, prompt_b, classify_schema, 350)
    print("RAW GEMINI CLASSIFY (B1):\n", raw_b1)
    obj_b = _try_parse_any(raw_b1)

    if not (obj_b and isinstance(obj_b, dict) and "matched_skills" in obj_b):
        raw_b2 = _call_gemini(client, prompt_b + "\nNO markdown. NO extra text.\n", classify_schema, 350)
        print("RAW GEMINI CLASSIFY (B2):\n", raw_b2)
        obj_b = _try_parse_any(raw_b2)

        # Repair fallback
        if not (obj_b and isinstance(obj_b, dict) and "matched_skills" in obj_b):
            repaired_b = _ensure_json_via_repair(client, raw_b2 or raw_b1, classify_schema)
            if repaired_b and "matched_skills" in repaired_b:
                obj_b = repaired_b

    if not (obj_b and isinstance(obj_b, dict) and "matched_skills" in obj_b):
        # fallback: return lists only, minimal evidence
        evidence = [{"skill": s, "category": "must_have", "status": "missing", "evidence": "No parse"} for s in must]
        return {
            "must_have_skills": must,
            "nice_to_have_skills": nice,
            "matched_skills": [],
            "missing_skills": must + nice,
            "skill_evidence": evidence,
            "improvement_suggestions": ["Re-run. Model returned non-JSON output."],
        }

    matched = _dedupe_str_list(obj_b.get("matched_skills", []))
    missing = _dedupe_str_list(obj_b.get("missing_skills", []))
    evidence = obj_b.get("skill_evidence", []) or []
    suggestions = _dedupe_str_list(obj_b.get("improvement_suggestions", []))[:4]

    return {
        "must_have_skills": must,
        "nice_to_have_skills": nice,
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_evidence": evidence,
        "improvement_suggestions": suggestions,
    }

# -------------------------------------------------
# Analyze endpoint: explainable deterministic scoring
# -------------------------------------------------

@app.post("/analyze")
def analyze_resume(payload: AnalyzeRequest):
    future = executor.submit(run_gemini, payload.resume_text, payload.job_text)

    try:
        result = future.result(timeout=30) or {}

        must = _dedupe_str_list(result.get("must_have_skills", []))
        nice = _dedupe_str_list(result.get("nice_to_have_skills", []))
        matched = _dedupe_str_list(result.get("matched_skills", []))
        missing = _dedupe_str_list(result.get("missing_skills", []))
        breakdown = result.get("skill_evidence", []) or []
        suggestions = _dedupe_str_list(result.get("improvement_suggestions", []))[:5]

        matched_set = {s.lower() for s in matched}

        must_total = len(must)
        nice_total = len(nice)
        must_matched = sum(1 for s in must if s.lower() in matched_set)
        nice_matched = sum(1 for s in nice if s.lower() in matched_set)

        weights = {"must_have": 0.8, "nice_to_have": 0.2}
        must_ratio = (must_matched / must_total) if must_total else 0.0
        nice_ratio = (nice_matched / nice_total) if nice_total else 0.0

        score = 100.0 * (weights["must_have"] * must_ratio + weights["nice_to_have"] * nice_ratio)
        score_int = int(round(min(max(score, 0.0), 100.0)))

        # if model failed to produce skills, fall back to heuristic score (never 0 unless input empty)
        if must_total == 0 and nice_total == 0:
            score_int = int(round(_heuristic_score(payload.resume_text, payload.job_text)))
            suggestions = suggestions or ["Model output was not JSON; showing estimated score. Retry."]

        out = {
            "match_score": score_int,
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
            "improvement_suggestions": suggestions,
        }

        print("RETURNING TO CLIENT:", out)
        return out

    except TimeoutError:
        est = int(round(_heuristic_score(payload.resume_text, payload.job_text)))
        out = {
            "match_score": est,
            "matched_skills": [],
            "missing_skills": [],
            "skill_breakdown": [],
            "score_explanation": {"error": "Timed out. Estimated score only; retry."},
            "improvement_suggestions": ["Timed out. Please retry for full breakdown."],
        }
        print("RETURNING TO CLIENT (TIMEOUT):", out)
        return out

    except Exception as e:
        est = int(round(_heuristic_score(payload.resume_text, payload.job_text)))
        out = {
            "match_score": est,
            "matched_skills": [],
            "missing_skills": [],
            "skill_breakdown": [],
            "score_explanation": {"error": f"Error: {str(e)}"},
            "improvement_suggestions": ["Temporary error. Please retry."],
        }
        print("RETURNING TO CLIENT (ERROR):", out)
        return out
