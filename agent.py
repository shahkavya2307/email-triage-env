import os
import time
import json
from openai import OpenAI
from env import Observation, Action

# --- THE GRADER PROXY FIX ---
# We explicitly check for Meta's injected variables to satisfy the validator bot.
if "API_BASE_URL" in os.environ and "API_KEY" in os.environ:
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
        timeout=120.0
    )
    # The proxy often expects a specific routing name. We catch it here.
    eval_model = os.environ.get("MODEL_NAME", os.environ.get("MODEL_ID", "gpt-3.5-turbo"))
else:
    # If the grader isn't here, fallback to our Hugging Face testing setup
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ.get("HF_TOKEN"),
        timeout=120.0
    )
    eval_model = "Qwen/Qwen2.5-72B-Instruct"
# ----------------------------

def get_agent_action(obs: Observation, history: str = "", max_retries: int = 3) -> Action:
    """Passes the observation and past feedback to the LLM."""
    
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
            response = client.chat.completions.create(
                model=eval_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            raw_text = response.choices[0].message.content.strip()
            
            # --- THE MAGIC CLEANER ---
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
            wait_time = 5 * (attempt + 1)
            print(f"⚠️ Proxy/API busy. Retrying in {wait_time} seconds... | Error: {e}")
            time.sleep(wait_time)
            
    print("🛑 Agent completely failed to reach the proxy. Defaulting to 'archive'.")
    return Action(decision="archive")
