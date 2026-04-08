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
    decision: Literal["spam", "archive", "reply", "escalate"] = Field(
        description="The triage action to take on the current email."
    )
    # NEW: Allow the agent to draft a response for Medium/Hard tasks
    reply_text: Optional[str] = Field(
        default=None, 
        description="If decision is 'reply', write the suggested response here. Otherwise, leave null."
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

        # The 'info' dict is perfect for storing hidden ground truth so
        # our external metric graders can access it later without the AI seeing it.
        info = {
            "ground_truth": email["ground_truth"],
            "is_correct": is_correct,
            "expected_keywords": email["expected_keywords"], # NEW: Added for metrics.py
            "agent_reply": action.reply_text                 # NEW: Added for metrics.py
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
    
    # Simulate the agent choosing "spam"
    test_action = Action(decision="spam")
    next_obs, reward, is_done, info_dict = test_env.step(test_action)
    
    print("\nAction Taken:", test_action.decision)
    print("Reward:", reward.value)
    print("Info (Hidden Truth):", info_dict)