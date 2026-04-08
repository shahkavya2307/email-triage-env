---
title: Email Triage Env
emoji: 📬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 📬 OpenM Environment: AI Email Triage

## 🏆 The Big Idea
We are moving AI from a simple "text generator" to a software agent that can do multi-step tasks correctly. To do that, AI needs a practice playground. 

This project is an **Obstacle Course for AI**. It is a standard Reinforcement Learning (RL) environment simulating a real-world customer support inbox. The AI must read incoming emails and sort them into four buckets, adapting its strategy based on strict company rules.

## 🧠 Why This Environment Works
This isn't a toy game like tic-tac-toe; it's a simulation of a real workplace automation problem. 
* **Stateful:** The inbox changes after every action.
* **Verifiable:** Success is grounded in objective truth (correct bucket, correct keywords).
* **Impossible to Cheat:** We penalize the AI if it tries to write a polite reply during a "server down" emergency. It must escalate and stay quiet.

## ⚙️ The Action Space
The agent can choose between four actions:
1. `spam`
2. `archive`
3. `reply` (Must include a drafted response)
4. `escalate` (Must NOT include a drafted response)

## 📊 The Curriculum (Scoring)
The environment scales from easy to hard, testing the AI's reasoning:
* **Task 1 (Easy - Sorting):** Did the AI put the email in the right bucket?
* **Task 2 (Medium - Writing):** If replying, did the AI draft a helpful response with the correct expected keywords (e.g., "refund", "sorry")?
* **Task 3 (Hard - Judgment):** In a severe emergency, the AI must choose `escalate` and remain completely silent. If it drafts a reply, it receives a strict penalty.

## 🚀 Run it Yourself (Docker)
This environment is fully containerized. To test the AI's baseline performance, simply run:
```bash
docker build -t email-triage .
docker run email-triage
```

✅ Run complete. Calculating scores against hardcoded rules...


🏆 Baseline AI Performance

📊 Final Results:
Task 1 (Basic Accuracy):    80.00%
Task 2 (Weighted Accuracy): 84.62%
Task 3 (Penalized Score):   70.00%


# ✨ "Wow" Features & Strategic Decisions 🧠

### 1️⃣ Safety First: Emergency Logic (Anti-Hallucination)
Most AI models are "people-pleasers." 🚨  
**Decision:** If a server is down, escalate & stay silent.  
**Why it Wins:** Shows AI Safety & Judgment, not just logic.

---

### 2️⃣ JSON Sandboxing with Pydantic 🛠️
AI loves rambling. We force strict JSON output.  
**Decision:** Pydantic blocks unexpected formats to prevent crashes.  
**Why it Wins:** Ensures Runtime Correctness & engineering maturity.

---

### 3️⃣ In-Context Feedback Loop ("Sticky Note" Method) 🔄
No heavy RL needed—just smart in-context learning.  
**Decision:** `metrics.py` writes a sticky note on rule violations; next AI task reads it.  
**Why it Wins:** AI learns instantly like a human, no supercomputer required.

---

### 4️⃣ Resilience Engineering: Error 429/503 Handling 🛡️
Demo-proof under stress.  
**Decision:** Retry loop + 13s API delay; tells AI to "take a deep breath" when rate-limited.  
**Why it Wins:** Reliable, reproducible demo without crashes.

---

## 👥 Contributors
- **[Bhumi N Deshpande](https://github.com/bhumindeshpande8-spec)** 
- **[Sejal Pednekar](https://github.com/Sejalp-18)** 
