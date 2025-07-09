import os
import sys
import yaml
import json
import logging
import torch
import uvicorn
import httpx
import pandas as pd
import numpy as np
import asyncio
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from joblib import load
from contextlib import asynccontextmanager

# --- Setup Paths and Logging ---
CONFIG_PATH = "config/config.yaml"
try:
    with open(CONFIG_PATH, 'r') as f: config = yaml.safe_load(f)
except FileNotFoundError:
    sys.exit(f"FATAL: Configuration file not found at {CONFIG_PATH}")

log_level = config.get("log_level", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', force=True)
logger = logging.getLogger("InferenceAPI")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from agent.models import ActorCritic
from data.processor import DataProcessor

# --- Global State ---
app_state = {"rl_model": None, "feature_scaler": None, "scaler_colnames": None,
             "feature_list": None, "config": config, "device": None}

# --- Lifespan Event Handler (Modern FastAPI) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block runs on startup
    logger.info("--- API Lifespan Startup: Loading artifacts... ---")
    cfg = app_state['config']
    app_state['device'] = torch.device("cpu") # Forcing CPU for API server stability
    logger.info(f"Using device: {app_state['device']}")

    data_paths = cfg['data_paths']
    project_root = cfg.get('project_root', '.')
    
    try:
        features_list_path = os.path.join(project_root, data_paths['features_list_path'])
        with open(features_list_path, 'r') as f: app_state['feature_list'] = json.load(f)
        
        # NOTE: These are loaded but not used directly if DataProcessor handles it internally
        app_state['feature_scaler'] = load(os.path.join(project_root, data_paths['scaler_params_path']))
        app_state['scaler_colnames'] = load(os.path.join(project_root, data_paths['scaler_colnames_path']))
        
        env_cfg = cfg['environment']; agent_cfg = cfg['agent']['actor_critic']
        n_actions = 1 + 2 * len(env_cfg.get('position_size_fractions', [0.125, 0.25]))
        input_dims = 4 + 1 + env_cfg['lookback_window'] * len(app_state['feature_list'])
        
        rl_model = ActorCritic(input_dims=input_dims, n_actions=n_actions, h1=agent_cfg['hidden_size1'], h2=agent_cfg['hidden_size2'])
        model_path = os.path.join(project_root, cfg['training_loop']['model_save_directory'], "ppo_trading_agent_best_val.pth")
        rl_model.load_state_dict(torch.load(model_path, map_location=app_state['device']))
        rl_model.to(app_state['device']); rl_model.eval()
        app_state['rl_model'] = rl_model
        logger.info(f"RL model and all artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL ERROR during artifact loading: {e}", exc_info=True)
        app_state['rl_model'] = None
    
    yield # The API is now running
    
    logger.info("--- API Lifespan Shutdown ---")

app = FastAPI(title="RL + LLM Trading Inference API", version="1.1.0", lifespan=lifespan)

def load_prompt_template(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

# --- Helper Functions ---
def get_latest_market_data_from_disk(config: dict) -> Dict[str, Dict[str, pd.DataFrame]]:
    raw_dir = config['data_paths']['raw_data_directory']
    symbols = config['producer']['symbols_to_fetch']
    timeframes = config['data_processing']['feature_engineer_timeframes']
    rows_to_fetch = config['environment']['lookback_window'] + 200
    all_data = {s: {} for s in symbols}
    
    for symbol in symbols:
        for tf in timeframes:
            filename = f"klines_{tf}.csv" 
            filepath = os.path.join(config.get("project_root", "."), raw_dir, symbol, filename)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                try: 
                    all_data[symbol][tf] = pd.read_csv(filepath).tail(rows_to_fetch).reset_index(drop=True)
                except Exception as e: 
                    logger.error(f"Could not read {filepath}: {e}")
                    all_data[symbol][tf] = pd.DataFrame()
            else: 
                logger.warning(f"Raw data file not found or empty: {filepath}")
                all_data[symbol][tf] = pd.DataFrame()
    return all_data

def format_df_for_prompt(df: pd.DataFrame, n_rows: int = 5) -> str:
    """Converts the tail of a DataFrame into a string for the LLM prompt."""
    if df.empty:
        return "No data available."
    cols_to_show = [col for col in df.columns if 'Timestamp' in col or 'Close_' in col or 'Volume_' in col or 'RSI' in col or 'BBW' in col]
    return df[cols_to_show].tail(n_rows).to_string()

async def query_ollama_llm(unscaled_df: pd.DataFrame, rl_decision: str, config: dict) -> Dict[str, Any]:
    llm_cfg = config['llm']
    data_summary = format_df_for_prompt(unscaled_df)

    prompt_template_path = llm_cfg['explanation_prompt_template']
    prompt_template = load_prompt_template(prompt_template_path)
    final_prompt = prompt_template.format(rl_decision=rl_decision, data_summary=data_summary)
    
    logger.info(f"Final prompt for LLM: {final_prompt}")

    payload = {
        "model": llm_cfg['ollama_model_name'],
        "messages": [{"role": "user", "content": final_prompt}],
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(llm_cfg['ollama_api_url'].replace("/generate", "/chat"), json=payload)
        response.raise_for_status()
        
        response_json = response.json()
        message_content = response_json.get('message', {}).get('content', '{}')
        response_data = json.loads(message_content)
        
        logger.info(f"Ollama raw response JSON: {response.json()}")
        explanation = response_data.get("explanation", "Error: Missing explanation from LLM.")
        return {"explanation": explanation}

    except Exception as e:
        logger.error(f"Error querying Ollama LLM: {e}", exc_info=True)
        return {"explanation": "An error occurred while generating the explanation."}

def construct_observation_from_df(processed_df: pd.DataFrame, feature_list: list, config: dict) -> np.ndarray:
    """Constructs the observation vector for the RL agent."""
    lookback = config['environment']['lookback_window']
    if len(processed_df) < lookback: return None
    
    df_for_obs = pd.DataFrame(index=processed_df.tail(lookback).index)
    for col in feature_list:
        df_for_obs[col] = processed_df[col] if col in processed_df.columns else 0
        
    feature_data = df_for_obs.values.flatten().astype(np.float32)
    dummy_portfolio_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    dummy_market_regime = np.array([0.0], dtype=np.float32)
    return np.concatenate([dummy_portfolio_state, dummy_market_regime, feature_data])

async def run_rl_inference(processed_df: pd.DataFrame, config: dict) -> str:
    """Runs inference on the RL model."""
    obs_vector = construct_observation_from_df(processed_df, app_state['feature_list'], config)
    if obs_vector is None: return "INSUFFICIENT_DATA"
    
    obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).to(app_state['device'])
    with torch.no_grad():
        action_tensor, _, _, _ = app_state['rl_model'].get_action_and_value(obs_tensor)
    rl_decision_idx = action_tensor.item()
    
    n_fractions = len(config['environment'].get('position_size_fractions', [0.125, 0.25]))
    if rl_decision_idx == 0: return "HOLD"
    elif 0 < rl_decision_idx <= n_fractions: return "LONG"
    else: return "SHORT"

# --- Main API Endpoint ---
@app.post("/get-decision", summary="Get RL + LLM Trading Analysis", tags=["Inference"])
async def get_decision():
    if not app_state.get("rl_model"):
        raise HTTPException(status_code=503, detail="RL model is not loaded. Check startup logs for errors.")
    
    cfg = app_state['config']
    primary_symbol = cfg['producer']['symbols_to_fetch'][0]
    
    latest_raw_data = get_latest_market_data_from_disk(cfg)
    
    primary_tf_data = latest_raw_data.get(primary_symbol, {}).get(cfg['data_processing']['primary_timeframe'])
    if primary_tf_data is None or primary_tf_data.empty:
        raise HTTPException(status_code=503, detail=f"Could not fetch latest primary data for {primary_symbol}. Ensure the data producer is running.")

    dp = DataProcessor(CONFIG_PATH)

    processed_dfs, unscaled_dfs = {}, {}
    for symbol, tf_data in latest_raw_data.items():
        if tf_data and all(not df.empty for df in tf_data.values()):
            try:
                proc_df, unscaled_df = dp.process_live_data(tf_data)
                if proc_df is not None and not proc_df.empty:
                    processed_dfs[symbol] = proc_df
                    unscaled_dfs[symbol] = unscaled_df
            except Exception as e:
                logger.error(f"Error processing live data for {symbol}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Data processing failed for symbol {symbol}.")

    if primary_symbol not in processed_dfs:
        raise HTTPException(status_code=500, detail="Primary symbol data processing failed.")
    
    # --- MODIFIED LOGIC: Run sequentially ---
    # 1. Get the RL agent's decision first.
    rl_decision = await run_rl_inference(processed_dfs[primary_symbol], cfg)

    # 2. Then, pass the unscaled data and the RL decision to the LLM to get an explanation.
    llm_analysis = await query_ollama_llm(unscaled_dfs[primary_symbol], rl_decision, cfg)
    
    final_output = {"decision": rl_decision, "llm": llm_analysis}
    return final_output

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)