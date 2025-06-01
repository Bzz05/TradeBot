import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os

from agent.ppo_agent import PPOAgent
from environment.trading_env import TradingEnv
from config import config

app = FastAPI()

# Define the request schema
class StatePayload(BaseModel):
    state: list[float]

# Load the model once at startup
@app.on_event("startup")
def load_model():
    global agent_model, device
    device = torch.device("cpu")
    
    obs_dim = config["obs_dim"]
    act_dim = config["act_dim"]
    agent_model = PPOAgent(obs_dim, act_dim).to(device)

    model_path = "ppo_trading_agent_final.pth"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file {model_path} not found")
    agent_model.load_state_dict(torch.load(model_path, map_location=device))
    agent_model.eval()

@app.post("/score")
def score(payload: StatePayload):
    state_list = payload.state
    if len(state_list) != config["obs_dim"]:
        raise HTTPException(status_code=400, detail=(
            f"Expected state dimension {config['obs_dim']}, got {len(state_list)}"
        ))
    try:
        st = torch.tensor(state_list, dtype=torch.float32).to(device).unsqueeze(0)
        with torch.no_grad():
            action_tensor = agent_model.select_action(st)
        action_int = action_tensor.item() if hasattr(action_tensor, "item") else int(action_tensor)
        return {"action": action_int}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))