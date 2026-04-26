import sys
import os
# This line tells the folder to look outside itself to find your env.py file!
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from env import EmailEnv, EmailAction

app = FastAPI()
office = EmailEnv(num_emails=10)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    obs = office.reset()
    return obs.model_dump()

@app.get("/state")
def state():
    obs = office.state()
    return obs.model_dump()

@app.post("/step")
def step(action: EmailAction):
    obs, reward, done, info = office.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.value,
        "done": done,
        "info": info
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
