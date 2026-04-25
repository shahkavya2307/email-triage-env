import os
import time
import json
from openai import OpenAI
from env import Observation, Action

# --- [WINNER'S UPGRADE: MANAGED SLIDING WINDOW] ---
# We store the notes in a list to prevent "Memory Bloat" and context errors.
sticky_notes = [] 

def update_memory(new_feedback: str):
    """Adds new feedback and keeps only the most recent 3 items[cite: 112]."""
    global sticky_notes
    if new_feedback and new_feedback.strip():
        sticky_notes.append(new_feedback.strip())
    
    # Rolling Window: ensures the model stays focused and inference stays fast[cite: 145].
    if len(sticky_notes) > 3:
        sticky_notes.pop(0) 

def clear_memory():
    """Wipes the sliding window for a fresh episode start[cite: 45, 179]."""
    global sticky_notes
    sticky_notes = []
# --------------------------------------------------

# --- THE GRADER PROXY FIX  ---
# Detects if we are running in Meta's evaluation environment or locally.
if "API_BASE_URL" in os.environ and "API_KEY" in os.environ:
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
        timeout=120.0
    )
    eval_model = os.environ.get("MODEL_NAME", os.environ.get("MODEL_ID", "gpt-3.5-turbo"))
else:
    # Fallback for local testing (Hugging Face Router)
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ.get("HF_TOKEN"),
        timeout=120.0
    )
    eval_model = "Qwen/Qwen2.5-72B-Instruct"
# ---------------------------------------------

def get_agent_action(obs: Observation, current_feedback: str = "", max_retries: int = 3) -> Action:
    """Passes observations and rolling feedback to the LLM[cite: 22, 23]."""
    
    # 1. Update memory with latest environment feedback[cite: 112, 118].
    update_memory(current_feedback)
    
    # 2. Format the "Sticky Notes" for the prompt[cite: 337].
    history_display = "\n    ".join([f"- {note}" for note in sticky_notes])
    feedback_section = f"PAST LESSONS (Correct your behavior based on these):\n    {history_display}" if sticky_notes else ""

    prompt = f"""
    You are an AI email triage agent. 
    
    RULES:
    1. Decide the best action: spam, archive, reply, escalate.
    2. If the action is 'reply', you MUST draft a polite, helpful response.
    3. If the action is 'escalate' (e.g., severe issues, server down), you MUST NOT write a reply. Leave it empty.
    
    {feedback_section}
    
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
                temperature=0.0  # Keep it objective and deterministic[cite: 19].
            )
            
            raw_text = response.choices[0].message.content.strip()
            
            # --- THE MAGIC CLEANER [cite: 335] ---
            # Handles models that ignore instructions and wrap JSON in markdown.
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()
            # ------------------------------------

            parsed_json = json.loads(raw_text)
            
            # --- WINNER'S SAFETY OVERRIDE [cite: 87, 333] ---
            # Force compliance with the "Silent Escalation" rule to stop reward hacking[cite: 98, 100].
            if parsed_json.get("decision") == "escalate":
                parsed_json["reply_text"] = None
                
            return Action(**parsed_json)
            
        except Exception as e:
            # Resilience Engineering: Handle API delays and rate limits[cite: 339, 340].
            wait_time = 5 * (attempt + 1)
            print(f"⚠️ API Busy. Retrying in {wait_time}s... | Error: {e}")
            time.sleep(wait_time)
            
    # Absolute Fallback to keep the loop running[cite: 113].
    print("🛑 Critical Failure: Defaulting to 'archive'.")
    return Action(decision="archive")
