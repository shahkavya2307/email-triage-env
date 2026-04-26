# 📬 OpenEnv: Building a Reliable AI Email Triage Agent

At the **Meta PyTorch OpenEnv Hackathon**, our team built **OpenEnv: AI Email Triage Agent** - a practical AI environment designed to train and evaluate autonomous agents on real-world email operations. Instead of only generating text, the objective was to build an agent that can make dependable decisions under constraints. 

The environment simulates a customer support inbox where the AI must read incoming emails and decide the correct next action. The agent can classify messages as **spam**, **archive** routine emails, **reply** with a context-aware response, **escalate** urgent incidents, or choose **needs_human_review** when confidence on response is low. Every output follows a strict JSON schema, making responses machine-readable and production-oriented. 

We built the system on top of the **OpenEnv framework** using custom environment, observation, and action classes. The inbox state updates after every action, creating a stateful simulation where decisions have measurable consequences. This allows the agent to be evaluated not just on language quality, but on operational correctness. 

A key focus was **AI safety and judgment**. For example, in critical scenarios such as a “production server down” email, the correct behavior is to immediately escalate and remain silent rather than generate a polite but harmful response. Incorrect responses were penalized heavily, ensuring the model learns when action is more important than conversation. 

To improve performance, we fine-tuned a base model using **Unsloth** and **Hugging Face TRL** on a synthetic dataset that mirrored real triage inputs and expected structured outputs. 

![Training Loss Curve](training_loss_plot.png)
*Our training loss rapidly converged, proving the agent quickly learned the required JSON formatting and triage constraints.*

We then added an in-context feedback loop where the environment provides targeted correction messages after mistakes, helping the model adapt during inference. 

This combination produced measurable gains: the average score improved from **31.16% (baseline)** to **43.72%**, a **12.56% increase** in task performance. 

OpenEnv demonstrates our belief that the future of AI lies not only in generating text, but in building reliable agents that can reason, follow rules, and operate safely in real workflows.

-TEAM BUILDCODERS