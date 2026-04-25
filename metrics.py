import re
import time
from typing import List, Dict
from openai import OpenAI

# ---------------------------------------------------------
# THE JUDGE SETUP (The Official Walkie-Talkie)
# ---------------------------------------------------------
client = OpenAI(
    api_key="YOUR_GEMINI_API_KEY", # ⚠️ REPLACE THIS!
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def llm_judge_reply(original_email: str, drafted_reply: str) -> float:
    """
    Acts as a strict teacher grading the drafted reply.
    """
    # If the AI didn't write anything, automatic zero.
    if not drafted_reply:
        return 0.0

    prompt = f"""
    You are a strict grading teacher. Read this customer support email:
    Customer's Email: '{original_email}'
    
    Now read the AI assistant's drafted reply:
    Drafted Reply: '{drafted_reply}'
    
    If the reply is polite, helpful, and addresses the customer's problem, output a score of 1.0. 
    If it is rude, unhelpful, or makes no sense, output 0.0. 
    Output ONLY the number. Do not say anything else.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-flash", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0 
            )
            
            raw_answer = response.choices[0].message.content
            # Safely extract only the number from the AI's response
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", raw_answer)
            
            if numbers:
                return float(numbers[0])
            else:
                return 0.0

        except Exception as e:
            print(f"Judge is busy. Retrying in 5 seconds... ({e})")
            time.sleep(5)
            
    return 0.0 

# ---------------------------------------------------------
# THE SCORING FUNCTIONS
# ---------------------------------------------------------

def task1_score(emails: List[Dict], actions: List[Dict]) -> float:
    """Easy Task: Just check if the action matches ground_truth."""
    correct = 0
    for email, action in zip(emails, actions):
        if action["decision"] == email["ground_truth"]:
            correct += 1
    return correct / len(emails)


def task2_score(emails: List[Dict], actions: List[Dict]) -> float:
    """
    Medium Task (UPGRADED):
    If the AI was supposed to reply, we call the LLM Judge to grade the written text!
    """
    score = 0
    total = 0

    for email, action in zip(emails, actions):
        weight = 2 if email["ground_truth"] in ["reply", "escalate"] else 1
        total += weight

        # Did the AI make the right decision?
        if action["decision"] == email["ground_truth"]:
            
            # THE HACKATHON UPGRADE: If it's a reply, use the LLM to judge the quality
            if action["decision"] == "reply":
                print(f"\nEvaluating Reply for: {email['text'][:30]}...")
                # The judge gives a score between 0.0 and 1.0
                reply_quality = llm_judge_reply(email["text"], action.get("drafted_reply", ""))
                
                # We multiply the quality by the weight (2 points max)
                score += (reply_quality * weight)
                print(f"Judge awarded: {reply_quality * weight} out of {weight} points")
            
            # If it's not a reply, they just get the full points for a correct decision
            else:
                score += weight

    return score / total


def task3_score(emails: List[Dict], actions: List[Dict]) -> float:
    """Hard Task: Penalize wrong actions."""
    score = 0
    max_score = len(emails) * 2

    for email, action in zip(emails, actions):
        if action["decision"] == email["ground_truth"]:
            score += 2
        else:
            score -= 1

    score = max(score, 0)
    return score / max_score

# ---------------------------------------------------------
# TEST HARNESS
# ---------------------------------------------------------
if __name__ == "__main__":
    # Notice how we added 'text' to the emails so the Judge has something to read!
    emails = [
        {"ground_truth": "spam", "text": "BUY CHEAP ROLEX NOW!!!"},
        {"ground_truth": "reply", "text": "I was charged twice for my shoes, please help!"},
        {"ground_truth": "archive", "text": "Weekly newsletter from HR."},
    ]

    # Notice how actions are now dictionaries containing the drafted text!
    actions = [
        {"decision": "spam", "drafted_reply": ""},
        {"decision": "reply", "drafted_reply": "I am so sorry about the double charge. I have issued a full refund right now."},
        {"decision": "archive", "drafted_reply": ""}
    ]

    print("Task1 Score:", task1_score(emails, actions))
    print("Task2 Score:", task2_score(emails, actions))
    print("Task3 Score:", task3_score(emails, actions))
