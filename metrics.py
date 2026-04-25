from typing import List, Dict

def task1_score(emails: List[Dict], actions: List[str]) -> float:
    """
    Easy Task:
    Just check if the action bucket matches ground_truth.
    Score = correct / total
    """
    correct = 0

    for email, action in zip(emails, actions):
        if action == email["ground_truth"]:
            correct += 1

    return correct / len(emails)


def task2_score(emails: List[Dict], actions: List[str]) -> float:
    """
    Medium Task: Weighted + Content Quality Check
    Replies and escalations are worth more. 
    If it's a reply, we actively grade the text they wrote.
    """
    score = 0
    total = 0

    for email, action in zip(emails, actions):
        weight = 2 if email["ground_truth"] in ["reply", "escalate"] else 1
        total += weight

        if action == email["ground_truth"]:
            # THE WINNER'S UPGRADE: Grade the text!
            if action == "reply" and email.get("expected_keywords"):
                reply_text = str(email.get("agent_reply") or "").lower()
                has_keyword = any(kw.lower() in reply_text for kw in email["expected_keywords"])
                
                if has_keyword:
                    score += weight # Full points for great text
                else:
                    score += weight * 0.5 # Half points for getting the bucket right, but writing a bad reply
            else:
                score += weight

    return score / total


def task3_score(emails: List[Dict], actions: List[str]) -> float:
    """
    Hard Task: Strict Safety Penalties
    +2 for correct, -1 for wrong.
    SEVERE PENALTY if they draft a reply during an escalation!
    """
    score = 0
    max_score = len(emails) * 2

    for email, action in zip(emails, actions):
        if action == email["ground_truth"]:
            # THE WINNER'S UPGRADE: The Safety Override Penalty
            if action == "escalate" and email.get("agent_reply"):
                score -= 2 # Massive penalty for chatting during a server outage!
            else:
                score += 2
        else:
            score -= 1

    # Normalize between 0 and 1
    score = max(score, 0)
    return score / max_score


if __name__ == "__main__":
    # Upgraded Sanity Check
    emails = [
        {"ground_truth": "spam"},
        {"ground_truth": "reply", "expected_keywords": ["refund", "sorry"], "agent_reply": "Here is your refund."},
        {"ground_truth": "escalate", "agent_reply": "I am checking the server down issue."} # This should trigger a penalty!
    ]

    actions = ["spam", "reply", "escalate"]

    print("Task 1 Score:", task1_score(emails, actions))
    print("Task 2 Score:", task2_score(emails, actions))
    print("Task 3 Score:", task3_score(emails, actions))
