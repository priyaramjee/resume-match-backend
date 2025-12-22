print(">>> Script started")

import google.generativeai as genai
import os
import json

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.5-flash")

resume_text = """
Senior Data Analytics Manager with 10+ years of experience.
Strong in SQL, Power BI, Databricks, stakeholder management.
Led global analytics initiatives and data platform migrations.
"""

job_text = """
We are looking for a Senior Analytics Manager.
Requirements:
- Strong SQL skills
- Experience with cloud analytics platforms
- Stakeholder communication
- Python is a plus
"""

prompt = f"""
You are an expert resume reviewer and career coach.

Resume:
<<<
{resume_text}
>>>

Job Description:
<<<
{job_text}
>>>

Tasks:
1. Give a match score from 0 to 100.
2. Identify missing skills or keywords.
3. Identify key strengths relevant to the role.
4. Suggest 3 to 5 concrete resume improvements.

CRITICAL INSTRUCTIONS:
- Respond with RAW JSON only
- Do NOT use markdown
- Do NOT wrap the response in ```json
- Do NOT add any explanatory text

JSON schema:
{{
  "match_score": number,
  "missing_skills": [string],
  "key_strengths": [string],
  "improvement_suggestions": [string]
}}
"""

def parse_json_safely(text):
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
        return json.loads(cleaned)
    except Exception as e:
        print("❌ JSON parsing failed")
        print(text)
        raise e

max_attempts = 2

for attempt in range(max_attempts):
    try:
        response = model.generate_content(prompt)
        result = parse_json_safely(response.text)
        break
    except Exception:
        if attempt == max_attempts - 1:
            raise
        print("🔁 Retrying...")

print("\n=== PARSED RESULT ===\n")
print(json.dumps(result, indent=2))
