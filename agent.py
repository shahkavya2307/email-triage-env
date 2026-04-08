import os
import time
import json
from openai import OpenAI
from env import Observation, Action

# Added a 120-second timeout. This tells our code to wait patiently 
# if the massive 72B model needs a minute to "wake up" from sleep mode.
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
    timeout=120.0 
)

def get_agent_action(obs: Observation, history: str = "", max_retries: int = 3) -> Action:
    """Passes the observation and past feedback to the HF Router."""
    
    prompt = f"""
    You are an AI email triage agent. 
    
    RULES:
    1. Decide the best action: spam, archive, reply, escalate.
    2. If the action is 'reply', you MUST draft a polite, helpful response.
    3. If the action is 'escalate' (e.g., severe issues, server down), you MUST NOT write a reply. Leave it empty.
    
    {history}
    
    CURRENT EMAIL:
    Subject: {obs.subject}
    Body: {obs.body}

    You MUST respond ONLY with valid JSON in this exact format. DO NOT USE MARKDOWN BACKTICKS. DO NOT ADD EXTRA TEXT.
    {{
        "decision": "spam", 
        "reply_text": "your drafted reply or null"
    }}
    """
    
    for attempt in range(max_retries):
        try:
            # We removed 'response_format' because it crashes the HF router.
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            raw_text = response.choices[0].message.content.strip()
            
            # --- THE MAGIC CLEANER ---
            # If the AI ignored our rule and added markdown, we strip it out safely
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            
            raw_text = raw_text.strip()
            # -------------------------

            parsed_json = json.loads(raw_text)
            return Action(**parsed_json)
            
        except Exception as e:
            error_msg = str(e)
            wait_time = 5 * (attempt + 1)
            print(f"⚠️ API busy or waking up. Retrying in {wait_time} seconds... | Error: {error_msg}")
            time.sleep(wait_time)
            
    print("🛑 Agent completely failed to reach the API. Defaulting to 'archive'.")
    return Action(decision="archive")