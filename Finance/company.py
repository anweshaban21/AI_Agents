import json
import anthropic
from dotenv import load_dotenv
load_dotenv(override=True)  
import os
client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

def detect_company_with_llm(user_query: str) -> dict:
    prompt = f"""
You are a financial query parser.

Extract the company name and stock ticker from the user's query.

Return ONLY valid JSON in this format:
{{
  "company_name": "...",
  "ticker": "..."
}}

Rules:
- If only company name is present, infer the most likely US stock ticker.
- If only ticker is present, keep company_name as best guess if possible.
- If nothing is found, return:
{{
  "company_name": null,
  "ticker": null
}}

User query: "{user_query}"
"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    text = response.content[0].text.strip()
    #print("RAW TEXT:", repr(text))

    # remove markdown fences if present
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)
#ans=detect_company_with_llm("What is the latest stock price of Apple Inc.?")
#print(ans)