import os
import time
import re
from dotenv import load_dotenv

# Load API keys from your .env file
load_dotenv()

from env import EmailEnv
from agent import get_agent_action, clear_memory, get_memory
from metrics import task1_score, task2_score, task3_score

def run_local_test():
    # WINNER'S TIP: Always start with a clean slate so old lessons 
    # from previous runs don't pollute the new test episode.
    clear_memory() 
    
    print("🚀 Starting Adaptive AI Test...\n" + "="*50)

    # 1. Dynamically find task names from openenv.yaml to stay flexible [cite: 419, 421]
    task_names = ["task1", "task2", "task3"]
    try:
        with open("openenv.yaml", "r") as f:
            content = f.read()
            found_ids = re.findall(r'-\s*id:\s*([A-Za-z0-9_-]+)', content)
            if len(found_ids) >= 3:
                task_names = found_ids[-3:]
    except Exception as e:
        print(f"Could not read yaml, using defaults: {e}")

    # 2. Initialize the environment [cite: 44, 45]
    env = EmailEnv(num_emails=15)
    obs = env.reset()
    
    actions_taken = []
    evaluated_emails = [] 
    step_rewards_history = [] 
    
    # Starting state for our In-Context Feedback Loop [cite: 337, 338]
    current_feedback = ""
    step_count = 0
    
    try:
        while not obs.is_done:
            print(f"📧 New Email: [{obs.subject}] {obs.body}")
            
            # --- THE FEEDBACK HANDSHAKE ---
            # 1. Pass the 'Sticky Note' from the LAST step to the agent [cite: 337]
            action = get_agent_action(obs, current_feedback=current_feedback)
            
            # 2. The environment executes the action and generates new feedback [cite: 46, 58]
            next_obs, reward, done, info = env.step(action)
            
            step_count += 1
            step_rewards_history.append(reward.value)
            
            # 3. Extract the high-signal feedback for the NEXT loop iteration [cite: 118, 125]
            current_feedback = info.get("feedback", "")
            
            # --- NEW SCORING DISPLAY ---
            conf = getattr(action, "confidence_score", 1.0)
            if action.decision == "needs_human_review":
                print(f"⚠️ Score: {reward.value}/1.0 - Sent to Human Review (Confidence was low: {conf:.2f}).")
            elif reward.value == 1.0:
                print(f"✅ Score: 1.0/1.0 - 100% Right! Good job on choosing '{action.decision}' (Confidence: {conf:.2f}).")
            else:
                print(f"❌ Score: {reward.value}/1.0 - Needs Improvement (Confidence: {conf:.2f}).")
                
            # --- SHOW GENERATED REPLY ---
            if action.decision == "reply" and getattr(action, "reply_text", None):
                print(f"💬 Generated Reply: {action.reply_text}")
                
            if current_feedback:
                print(f"📝 STICKY NOTE ADDED: {current_feedback}")
            
            # Store data for final scoring
            actions_taken.append(action.decision.lower())
            evaluated_emails.append(info)
            
            obs = next_obs
            
            # Resilience Engineering: Respect rate limits during evaluation [cite: 110, 339]
            if not obs.is_done:
                time.sleep(13)

    except Exception as e:
        print(f"⚠️ Loop interrupted by API proxy, saving current progress... | {e}")
    
    finally:
        # 3. Calculate final results using your multi-tier metrics [cite: 81, 89]
        t1, t2, t3 = 0.0, 0.0, 0.0
        if len(actions_taken) > 0:
            t1 = task1_score(evaluated_emails, actions_taken)
            t2 = task2_score(evaluated_emails, actions_taken)
            t3 = task3_score(evaluated_emails, actions_taken)
        
        # --- THE SCORE CLAMPER ---
        # Nudges perfect or zero scores to ensure they are float-compatible with graders.
        def clamp_score(s):
            if s >= 1.0: return 0.99
            if s <= 0.0: return 0.01
            return float(s)
            
        t1, t2, t3 = clamp_score(t1), clamp_score(t2), clamp_score(t3)
        
        # 4. Format the final output for the Meta Log Parsers [cite: 183, 190]
        print("\n🤖 Sending formatted logs to the Meta Grader...", flush=True)
        
        tasks = [
            (task_names[0], t1),
            (task_names[1], t2),
            (task_names[2], t3)
        ]
        
        for name, final_score in tasks:
            print(f"[START] task={name}", flush=True)
            for i, r in enumerate(step_rewards_history):
                print(f"[STEP] step={i+1} reward={r}", flush=True)
            print(f"[END] task={name} score={final_score} steps={step_count}", flush=True)

        # --- NEW MARKDOWN REPORT GENERATOR ---
        try:
            current_avg = (t1 + t2 + t3) / 3.0 * 100
            previous_avg = None
            
            report_path = "triage_report.md"
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    match = re.search(r"\*\*Current Average Score:\*\* ([\d\.]+)%", content)
                    if match:
                        previous_avg = float(match.group(1))
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# 📊 AI Email Triage Report\n\n")
                f.write(f"**Total Emails Processed:** {step_count}\n\n")
                
                f.write("## 🏆 Performance Comparison\n")
                if previous_avg is not None:
                    f.write(f"- **Previous Average Score:** {previous_avg:.2f}%\n")
                    diff = current_avg - previous_avg
                    trend = "📈 Improved" if diff > 0 else ("📉 Regressed" if diff < 0 else "➖ Unchanged")
                    f.write(f"- **Current Average Score:** {current_avg:.2f}% ({trend} by {abs(diff):.2f}%)\n\n")
                else:
                    f.write(f"- **Current Average Score:** {current_avg:.2f}%\n\n")
                
                f.write("### Task Breakdown\n")
                f.write(f"- Task 1 (Bucketing): {t1*100:.2f}%\n")
                f.write(f"- Task 2 (Replies): {t2*100:.2f}%\n")
                f.write(f"- Task 3 (Escalation Safety): {t3*100:.2f}%\n\n")
                
                f.write("## 📝 Final Lessons Learned (Agent's Memory)\n")
                memory = get_memory()
                if memory:
                    for note in memory:
                        f.write(f"- {note}\n")
                else:
                    f.write("- *(No lessons learned this run! Perfect score?)*\n")
                
                f.write("\n## ❌ Mistakes & Human Reviews\n")
                mistakes = False
                for i, (info, decision) in enumerate(zip(evaluated_emails, actions_taken)):
                    if info.get("feedback"):
                        mistakes = True
                        f.write(f"**Step {i+1}:**\n")
                        f.write(f"- **AI Decision:** `{decision}`\n")
                        f.write(f"- **Ground Truth:** `{info.get('ground_truth')}`\n")
                        f.write(f"- **Feedback:** {info.get('feedback')}\n\n")
                
                if not mistakes:
                    f.write("*Flawless victory! No mistakes made.*\n")
                    
            print(f"\n📄 Saved detailed report to {report_path}")
            
        except Exception as report_err:
            print(f"⚠️ Failed to generate report: {report_err}")

if __name__ == "__main__":
    run_local_test()
