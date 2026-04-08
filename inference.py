# main.py
import os
import time
from dotenv import load_dotenv

load_dotenv()

from env import EmailEnv
from agent import get_agent_action
from metrics import task1_score, task2_score, task3_score

def run_local_test():
    print("🚀 Starting Adaptive AI Test...\n" + "="*50)
    
    env = EmailEnv(num_emails=10)
    obs = env.reset()
    
    actions_taken = []
    evaluated_emails = [] 
    
    # NEW: This string will grow and act as the AI's memory
    learning_history = "PAST PERFORMANCE FEEDBACK:\n"
    
    while not obs.is_done:
        print(f"📧 New Email: [{obs.subject}] {obs.body}")
        
        # 1. Agent decides based on the email AND its past history
        # We only pass history if we actually have some
        current_history = learning_history if len(actions_taken) > 0 else ""
        action = get_agent_action(obs, history=current_history)
        
        # 2. Environment steps forward
        next_obs, reward, done, info = env.step(action)
        
        print(f"🤖 Agent Decision : {action.decision.upper()} (Ground Truth: {info['ground_truth'].upper()})")
        if action.reply_text:
            print(f"📝 Drafted Reply  : {action.reply_text}")
            
        print(f"⭐️ Step Reward    : {reward.value}")
        print("-" * 50)
        
        # --- NEW: Generate Feedback for the Agent's Memory ---
        feedback = f"- Email: '{obs.subject}' | You chose: {action.decision.upper()} | Correct Action was: {info['ground_truth'].upper()}"
        
        # Add hints if it failed Medium/Hard task rules
        if action.decision == "reply" and info['ground_truth'] == "reply":
            missing = [kw for kw in info['expected_keywords'] if kw not in str(action.reply_text).lower()]
            if missing:
                feedback += f" -> WARNING: Your reply was missing these concepts: {missing}"
        elif action.decision == "escalate" and info['ground_truth'] == "escalate":
            if action.reply_text:
                feedback += " -> PENALTY: You broke Rule 3! Never draft a reply for escalated emails."
                
        # Append to the growing memory string
        learning_history += feedback + "\n"
        # ------------------------------------------------------
        
        # 3. Store data for final scoring
        # 3. Store data for final scoring
        actions_taken.append(action.decision.lower())
        evaluated_emails.append(info)
        
        obs = next_obs
        
        # 4. Respect Free Tier API Quotas
        if not obs.is_done:
            print("⏳ Pausing for 13 seconds to avoid API rate limits...\n")
            time.sleep(13)

    print("\n✅ Run complete. Calculating scores against hardcoded rules...")
    
    t1 = task1_score(evaluated_emails, actions_taken)
    t2 = task2_score(evaluated_emails, actions_taken)
    t3 = task3_score(evaluated_emails, actions_taken)
    
    print("\n📊 Final Results:")
    print(f"Task 1 (Basic Accuracy):    {t1:.2%}")
    print(f"Task 2 (Weighted Accuracy): {t2:.2%}")
    print(f"Task 3 (Penalized Score):   {t3:.2%}")

if __name__ == "__main__":
    run_local_test()