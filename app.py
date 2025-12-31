from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
import json

from google import genai
from google.genai.types import GenerateContentConfig, Schema

# -------------------------------------------------
# App setup
# -------------------------------------------------

app = FastAPI(title="Resume Match API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for MVP
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
# Gemini runner (schema JSON + retry on bad JSON)
# -------------------------------------------------

def run_gemini(resume_text: str, job_text: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    schema = Schema(
        type="object",
        properties={
            "match_score": Schema(type="number"),
            "missing_skills": Schema(type="array", items=Schema(type="string")),
            "key_strengths": Schema(type="array", items=Schema(type="string")),
            "improvement_suggestions": Schema(type="array", items=Schema(type="string")),
        },
        required=[
            "match_score",
            "missing_skills",
            "key_strengths",
            "improvement_suggestions",
        ],
    )

    def call_model(extra_instruction: str = "") -> str:
        prompt = (
            "You are an expert resume reviewer.\n\n"
            "Compare the resume and the job description.\n\n"
            "Return JSON ONLY.\n"
            "Limits:\n"
            "- missing_skills: max 5 items\n"
            "- key_strengths: max 4 items\n"
            "- improvement_suggestions: max 4 items\n"
            "- Each item max 8 words\n"
            + extra_instruction +
            "\n\n"
            f"Resume:\n{resume_text}\n\n"
            f"Job Description:\n{job_text}\n"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=500,
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

    # Attempt 1
    raw = call_model()
    print("RAW GEMINI OUTPUT (1):\n", raw)

    try:
        return json.loads(raw)
    except Exception:
        # Attempt 2 (tighter)
        raw = call_model("\nIMPORTANT: Output must be a complete JSON object. Do not stop early.\n")
        print("RAW GEMINI OUTPUT (2):\n", raw)
        return json.loads(raw)

# -------------------------------------------------
# Analyze endpoint (GUARANTEED RESPONSE)
# -------------------------------------------------

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
