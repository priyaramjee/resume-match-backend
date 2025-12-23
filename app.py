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
    Extract the first JSON object found in text.
    Handles markdown fences and extra commentary.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Gemini output")
    return json.loads(match.group())

# -------------------------------------------------
# Gemini runner (OFF main thread, SAFE)
# -------------------------------------------------

def run_gemini(resume_text: str, job_text: str) -> dict:
