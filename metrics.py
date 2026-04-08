from typing import List, Dict


def task1_score(emails: List[Dict], actions: List[str]) -> float:
    """
    Easy Task:
    Just check if the action matches ground_truth.
    Score = correct / total
    """
    correct = 0

    for email, action in zip(emails, actions):
        if action == email["ground_truth"]:
            correct += 1

    return correct / len(emails)


def task2_score(emails: List[Dict], actions: List[str]) -> float:
    """
    Medium Task:
    Only care about important emails:
    reply and escalate are more important.
    They give 2 points. Others give 1 point.
    """
    score = 0
    total = 0

    for email, action in zip(emails, actions):
        weight = 2 if email["ground_truth"] in ["reply", "escalate"] else 1
        total += weight

        if action == email["ground_truth"]:
            score += weight

    return score / total


def task3_score(emails: List[Dict], actions: List[str]) -> float:
    """
    Hard Task:
    Penalize wrong actions.
    +2 for correct, -1 for wrong
    """
    score = 0
    max_score = len(emails) * 2

    for email, action in zip(emails, actions):
        if action == email["ground_truth"]:
            score += 2
        else:
            score -= 1

    # Normalize between 0 and 1
    score = max(score, 0)
    return score / max_score

if __name__ == "__main__":
    emails = [
        {"ground_truth": "spam"},
        {"ground_truth": "reply"},
        {"ground_truth": "archive"},
    ]

    actions = ["spam", "archive", "archive"]

    print("Task1:", task1_score(emails, actions))
    print("Task2:", task2_score(emails, actions))
    print("Task3:", task3_score(emails, actions))