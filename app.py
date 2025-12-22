from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
import json

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = FastAPI(title="Resume Matcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for MVP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

# -------------------------------------------------
# Request / Response Schemas
# -------------------------------------------------
class AnalyzeRequest(BaseModel):
    resume_text: str
    job_text: str

class AnalyzeResponse(BaseModel):
    match_score: int
    missing_skills: list[str]
    key_strengths: list[str]
    improvement_suggestions: list[str]

# -------------------------------------------------
# Helper: Normalize skills (remove adjectives)
# -------------------------------------------------
def normalize_skills(skills: list[str]) -> list[str]:
    """
    Removes adjectives or non-skill tokens from missing_skills.
    """
    STOPWORDS = {
        "strong", "advanced", "experienced", "expert",
        "good", "solid", "hands-on", "proficient"
    }

    cleaned = []
    for skill in skills:
        s = skill.lower().strip()
        if s in STOPWORDS:
            continue
        cleaned.append(skill)

    return cleaned

# -------------------------------------------------
# Helper: Safe JSON parsing
# -------------------------------------------------
def parse_json_safely(text: str):
    """
    Ensures raw JSON can be parsed even if the model
    accidentally adds formatting.
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]

    return json.loads(cleaned)

# -------------------------------------------------
# Core API Endpoint
# -------------------------------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_resume(req: AnalyzeRequest):
    prompt = f"""
You are an expert resume reviewer and career coach.

Resume:
<<<
{req.resume_text}
>>>

Job Description:
<<<
{req.job_text}
>>>

Tasks:
1. Give a match score from 0 to 100.
2. Identify missing skills or keywords.
3. Identify key strengths relevant to the role.
4. Suggest 3 to 5 concrete resume improvements.

CRITICAL INSTRUCTIONS:
- Respond with RAW JSON only
- Do NOT use markdown
- Do NOT add explanations
- Missing skills must be concrete nouns or technologies
- Do NOT list adjectives (e.g. strong, advanced) as skills
- If a skill exists but lacks depth, suggest improvement instead

JSON schema:
{{
  "match_score": number,
  "missing_skills": [string],
  "key_strengths": [string],
  "improvement_suggestions": [string]
}}
"""

    max_attempts = 2

    for attempt in range(max_attempts):
        try:
            response = model.generate_content(prompt)
            result = parse_json_safely(response.text)

            # Normalize skills
            result["missing_skills"] = normalize_skills(
                result.get("missing_skills", [])
            )

            return result

        except Exception as e:
            if attempt == max_attempts - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM processing failed: {str(e)}"
                )
