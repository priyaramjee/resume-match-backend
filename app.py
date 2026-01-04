import os
import json
import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ----------------------------
# FastAPI setup
# ----------------------------
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
    return {"version": "skills-v4-parsed+retry+shared-client"}


# ----------------------------
# Gemini schemas (Pydantic)
# ----------------------------
class SkillLists(BaseModel):
    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)

class EvidenceItem(BaseModel):
    skill: str
    category: str  # "must_have"
    status: str    # "matched"|"missing"
    evidence: str  # <= 10 words

class ClassifyResult(BaseModel):
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    skill_evidence: List[EvidenceItem] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)


# ----------------------------
# Global Gemini client (async)
# ----------------------------
CLIENT: Optional[genai.Client] = None
ACLient: Optional[Any] = None  # async client

PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.0-flash"  # often steadier for strict extraction

@app.on_event("startup")
async def startup_event():
    global CLIENT, AClient
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    # Tip: you can also set api_version via http_options if you want stable endpoints.
    # See SDK docs for http_options usage. :contentReference[oaicite:3]{index=3}
    CLIENT = genai.Client(api_key=api_key)
    AClient = CLIENT.aio

@app.on_event("shutdown")
async def shutdown_event():
    global CLIENT, AClient
    try:
        if AClient is not None:
            await AClient.aclose()
    finally:
        AClient = None
        CLIENT = None


# ----------------------------
# Utilities
# ----------------------------
def _dedupe(seq: List[str]) -> List[str]:
    out, seen = [], set()
    for x in seq or []:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out

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


async def _gen_json(
    prompt: str,
    schema_model: Any,
    max_tokens: int,
    timeout_s: float,
    model: str,
) -> Optional[Any]:
    """
    Uses JSON mode + schema, and relies on response.parsed (no regex repair).
    Retries only when parsed/text is empty.
    """
    assert AClient is not None

    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        response_schema=schema_model,
    )

    # Small retry loop for "empty candidate" events reported in SDK issues.
    # :contentReference[oaicite:4]{index=4}
    for attempt in range(3):
        try:
            resp = await asyncio.wait_for(
                AClient.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=cfg,
                ),
                timeout=timeout_s,
            )

            parsed = getattr(resp, "parsed", None)
            if parsed is not None:
                return parsed

            # If parsed is missing but text exists, try a strict json load.
            txt = (getattr(resp, "text", None) or "").strip()
            if txt:
                try:
                    return schema_model.model_validate_json(txt)
                except Exception:
                    pass

        except asyncio.TimeoutError:
            # Let retries happen; caller can decide fallback behavior.
            pass
        except Exception:
            # transient errors: retry
            pass

        # jitter backoff
        await asyncio.sleep(0.15 * (2 ** attempt) + random.random() * 0.05)

    return None


async def run_gemini(resume_text: str, job_text: str) -> Dict[str, Any]:
    # Step A: skill list
    prompt_a = (
        "Return JSON ONLY.\n"
        "Extract skills as short nouns (1-3 words). Avoid long phrases.\n"
        "Output exactly two keys: must_have_skills, nice_to_have_skills.\n"
        "Limits: must_have_skills 5-8 items; nice_to_have_skills 0-4 items.\n\n"
        f"JOB DESCRIPTION:\n{job_text}\n"
    )

    lists = await _gen_json(
        prompt=prompt_a,
        schema_model=SkillLists,
        max_tokens=220,
        timeout_s=12.0,
        model=PRIMARY_MODEL,
    )

    # Fallback model if 2.5-flash returns empty/None
    if lists is None:
        lists = await _gen_json(
            prompt=prompt_a,
            schema_model=SkillLists,
            max_tokens=220,
            timeout_s=12.0,
            model=FALLBACK_MODEL,
        )

    if lists is None:
        return {
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "matched_skills": [],
            "missing_skills": [],
            "skill_evidence": [],
            "improvement_suggestions": ["Temporary model issue. Please try again."],
        }

    must = _dedupe(lists.must_have_skills)
    nice = _dedupe(lists.nice_to_have_skills)

    # Step B: classify
    prompt_b = (
        "Return JSON ONLY.\n"
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

    classified = await _gen_json(
        prompt=prompt_b,
        schema_model=ClassifyResult,
        max_tokens=350,
        timeout_s=14.0,
        model=PRIMARY_MODEL,
    )

    if classified is None:
        classified = await _gen_json(
            prompt=prompt_b,
            schema_model=ClassifyResult,
            max_tokens=350,
            timeout_s=14.0,
            model=FALLBACK_MODEL,
        )

    if classified is None:
        evidence = [
            {"skill": s, "category": "must_have", "status": "missing", "evidence": "No parse"}
            for s in must
        ]
        return {
            "must_have_skills": must,
            "nice_to_have_skills": nice,
            "matched_skills": [],
            "missing_skills": must + nice,
            "skill_evidence": evidence,
            "improvement_suggestions": ["Re-run. Model returned non-JSON/empty output."],
        }

    return {
        "must_have_skills": must,
        "nice_to_have_skills": nice,
        "matched_skills": _dedupe(classified.matched_skills),
        "missing_skills": _dedupe(classified.missing_skills),
        "skill_evidence": [e.model_dump() for e in (classified.skill_evidence or [])],
        "improvement_suggestions": _
