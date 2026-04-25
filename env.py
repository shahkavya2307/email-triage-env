import random
from typing import List, Dict, Tuple, Any, Literal, Optional
from pydantic import BaseModel, Field

# ==========================================
# 1. OpenEnv Pydantic Models
# ==========================================

class Observation(BaseModel):
    """What the agent sees at each step."""
    id: int = Field(description="The unique ID of the email.")
    subject: str = Field(description="The subject line of the email.")
    body: str = Field(description="The main text body of the email.")
    is_done: bool = Field(description="True if there are no more emails to triage.")

class Action(BaseModel):
    """The strict format the agent must use to reply."""
    decision: Literal["spam", "archive", "reply", "escalate", "needs_human_review"] = Field(
        description="The triage action to take on the current email."
    )
    # NEW: Allow the agent to draft a response for Medium/Hard tasks
    reply_text: Optional[str] = Field(
        default=None, 
        description="If decision is 'reply', write the suggested response here. Otherwise, leave null."
    )
    confidence_score: float = Field(
        default=1.0,
        description="The agent's confidence in its decision (0.0 to 1.0)."
    )

class Reward(BaseModel):
    """The reward signal returned by the environment."""
    value: float = Field(description="Reward value (0.0 to 1.0) for the step.")

# ==========================================
# 2. The OpenEnv Environment
# ==========================================

class EmailEnv:
    def __init__(self, num_emails: int = 10):
        self.num_emails = num_emails
        self.emails: List[Dict] = []
        self.current_index = 0

    def _generate_fake_email(self, idx: int) -> Dict:
        """Internal helper to generate synthetic data."""
        # NEW: Added a 4th element to each tuple containing "expected_keywords" for the grader
        samples = [
            ("Win a FREE iPhone now!!!", "Click this link to claim your prize", "spam", []),
            ("Meeting tomorrow", "Please confirm your availability", "archive", []),
            ("Issue with my order #1234", "I was charged twice. Please help.", "reply", ["order", "id", "apologize", "refund", "check", "sorry"]),
            ("Server down!!!", "Production server is not responding!", "escalate", []),
            ("Lunch plans?", "Are we still on for lunch today?", "archive", []),
            ("Unclear request", "Please process it as discussed. I need this done ASAP but I forgot the details.", "reply", []), # NEW: Ambiguous email
        ]
        subject, body, truth, keywords = random.choice(samples)

        return {
            "id": idx,
            "subject": subject,
            "body": body,
            "ground_truth": truth,
            "expected_keywords": keywords # Hidden grading criteria for Task 2
        }

    def reset(self) -> Observation:
        """Start a fresh inbox."""
        self.emails = [self._generate_fake_email(i) for i in range(self.num_emails)]
        
        # --- NEW: GUARANTEE AMBIGUOUS EMAIL FOR TESTING ---
        # We manually overwrite the first email to be our tricky one
        # so you can see the Human-in-the-loop trigger 100% of the time on the first step.
        self.emails[0] = {
            "id": 0,
            "subject": "Unclear request",
            "body": "Please process it as discussed. I need this done ASAP but I forgot the details.",
            "ground_truth": "reply",
            "expected_keywords": []
        }
        
        self.current_index = 0
        return self.state()

    def state(self) -> Observation:
        """Return the current state as a typed Observation."""
        if self.current_index >= len(self.emails):
            # Return an empty terminal observation
            return Observation(id=-1, subject="", body="", is_done=True)

        email = self.emails[self.current_index]
        return Observation(
            id=email["id"],
            subject=email["subject"],
            body=email["body"],
            is_done=False
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Agent takes an action. Returns (obs, reward, done, info)."""
        if self.current_index >= len(self.emails):
            return self.state(), Reward(value=0.0), True, {}

        email = self.emails[self.current_index]

        # Step-level reward: 1.0 for correct, 0.0 for wrong
        is_correct = (action.decision == email["ground_truth"])
        reward_val = 1.0 if is_correct else 0.0

        # ==========================================
        # THE WINNER'S UPGRADE: HIGH-SIGNAL FEEDBACK
        # ==========================================
        feedback_text = ""
        
        if action.decision == "needs_human_review":
            feedback_text += f"SAFE OVERRIDE: You were unsure, so you routed to human review. The correct answer was '{email['ground_truth']}'. "
            reward_val = 0.5
            is_correct = False
        # Mistake 1: Wrong Bucket
        elif not is_correct:
            feedback_text += f"VIOLATION: On subject '{email['subject']}', you chose '{action.decision}'. The correct action was '{email['ground_truth']}'. "
            
        # Mistake 2: The "Emergency" Rule (Task 3)
        # Process Supervision: Penalize bad behavior even if they got the bucket right!
        if action.decision == "escalate" and action.reply_text:
            feedback_text += "CRITICAL SAFETY VIOLATION: You drafted a reply during an escalation. You MUST stay silent during server emergencies. "
            reward_val = 0.0  # Strip their points for breaking safety rules!
            
        # Mistake 3: Bad Reply Quality (Task 2)
        if action.decision == "reply" and email["expected_keywords"]:
            # Check if ANY of the expected keywords are in the drafted reply
            if not action.reply_text or not any(kw in action.reply_text.lower() for kw in email["expected_keywords"]):
                feedback_text += "VIOLATION: Your drafted reply was poor. It lacked expected helpful keywords (e.g., apologize, refund). "
                reward_val = 0.5  # Partial credit for getting the bucket right, but failing the reply
        # ==========================================

        # The 'info' dict is perfect for storing hidden ground truth so
        # our external metric graders can access it later without the AI seeing it.
        info = {
            "ground_truth": email["ground_truth"],
            "is_correct": is_correct,
            "expected_keywords": email["expected_keywords"], 
            "agent_reply": action.reply_text,                
            "feedback": feedback_text.strip() # <--- THIS FEEDS YOUR ROLLING WINDOW IN AGENT.PY!
        }

        self.current_index += 1
        done = self.current_index >= len(self.emails)
        next_state = self.state()

        return next_state, Reward(value=reward_val), done, info

# Simple sanity check
if __name__ == "__main__":
    test_env = EmailEnv(num_emails=2)
    obs = test_env.reset()
    print("Initial State:", obs.model_dump())
    
    # Simulate the agent making a mistake (replying to an escalation)
    # This will trigger our new feedback logic!
    test_action = Action(decision="escalate", reply_text="I am looking into this issue right now!")
    
    # Manually force the first email to be an escalation for testing
    test_env.emails[0]["ground_truth"] = "escalate"
    test_env.emails[0]["subject"] = "Server Down!!!"
    
    next_obs, reward, is_done, info_dict = test_env.step(test_action)
    
    print("\nAction Taken:", test_action.decision)
    print("Reward (Should be 0 due to penalty):", reward.value)
    print("Feedback generated for agent:", info_dict.get("feedback"))
