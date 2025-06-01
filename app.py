# app.py
import logging
import os
import yaml
import torch
import uvicorn # For programmatic startup
from fastapi import FastAPI, HTTPException
from typing import Optional, List # Added List for type hinting if needed later
import numpy as np # Added for np.array if observation construction was here
import json # Added for loading features list

# --- Add src to Python path ---
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming app.py is in the project root, and 'src' is a subdirectory.
src_path = os.path.join(current_script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from agent.models import ActorCritic # Your model class
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import ActorCritic from agents.models.")
    print(f"Ensure src/agents/models.py exists and src/ and src/agents/ have __init__.py files.")
    print(f"Python search path (sys.path) includes: {src_path}")
    print(f"ImportError: {e}")
    ActorCritic = None 
    # We'll let the app start and report the error via /model-status

# --- Configuration ---
# This app will now use the main config file, expecting it at this path
CONFIG_PATH = "config/config.yaml" 

# --- Global Variables ---
model = None # To store the loaded model instance
app_config_loaded: Optional[dict] = None # To store the loaded configuration
model_loaded_successfully: bool = False
model_loading_error: Optional[str] = None
device: Optional[torch.device] = None
# We don't need input_dims_global here as it's calculated within startup

# --- Logging Setup ---
# Logging level will be set from config in startup_event
logger = logging.getLogger(__name__)


# --- FastAPI App Initialization & Lifespan Events ---
app = FastAPI(
    title="RL Trading Agent Status API (using main config)",
    description="Checks health and if the trained model (specified in config/config.yaml) can be loaded.",
    version="0.2.0"
)

@app.on_event("startup")
async def startup_event():
    """Load configuration and attempt to load the model on application startup."""
    global model, app_config_loaded, model_loaded_successfully, model_loading_error, device
    
    if ActorCritic is None:
        model_loaded_successfully = False
        model_loading_error = "ActorCritic class could not be imported. Check src/agents/models.py and imports."
        logging.basicConfig(level="ERROR") # Basic logging if config can't be read
        logger.error(model_loading_error)
        return

    try:
        if not os.path.exists(CONFIG_PATH):
            model_loaded_successfully = False
            model_loading_error = f"Main configuration file not found at {CONFIG_PATH}"
            logging.basicConfig(level="ERROR")
            logger.error(model_loading_error)
            return

        with open(CONFIG_PATH, 'r') as f:
            app_config_loaded = yaml.safe_load(f) # Store loaded config globally
        if not app_config_loaded:
            model_loaded_successfully = False
            model_loading_error = f"Main config file {CONFIG_PATH} is empty or invalid."
            logging.basicConfig(level="ERROR")
            logger.error(model_loading_error)
            return
        
        log_level_from_config = app_config_loaded.get("logging_level", "INFO").upper()
        logging.basicConfig(level=getattr(logging, log_level_from_config, logging.INFO),
                            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', force=True)
        logger.info(f"FastAPI application startup: Attempting to load model using {CONFIG_PATH}...")


        data_paths_cfg = app_config_loaded.get('data_paths', {})
        env_cfg = app_config_loaded.get('environment', {})
        agent_cfg = app_config_loaded.get('agent', {})
        actor_critic_cfg = agent_cfg.get('actor_critic', {})
        loop_cfg = app_config_loaded.get('training_loop', {})

        # Get model path
        # Prefer model_save_directory from training_loop, then data_paths, then default
        model_dir_rel = loop_cfg.get("model_save_directory", 
                                     data_paths_cfg.get("model_save_directory", 
                                                        os.path.join("src", "models", "saved_models", "PPO")))
        # Assuming app.py is at project root
        project_root = os.path.dirname(os.path.abspath(__file__)) 
        model_dir_abs = os.path.join(project_root, model_dir_rel) if not os.path.isabs(model_dir_rel) else model_dir_rel
        model_filename = "ppo_trading_agent_best_val.pth"
        model_path_abs = os.path.join(model_dir_abs, model_filename)

        if not os.path.exists(model_path_abs):
            model_loaded_successfully = False
            model_loading_error = f"Model file not found at resolved path: {model_path_abs}"
            logger.error(model_loading_error)
            return

        # Get features list path to calculate input_dims
        features_list_path_rel = data_paths_cfg.get("features_list_path", os.path.join("data","historical_processed","final_agent_features.json"))
        features_list_path_abs = os.path.join(project_root, features_list_path_rel) if not os.path.isabs(features_list_path_rel) else features_list_path_rel

        if not os.path.exists(features_list_path_abs):
            model_loaded_successfully = False
            model_loading_error = f"Agent features list not found at {features_list_path_abs}. Needed for input_dims."
            logger.error(model_loading_error)
            return
        with open(features_list_path_abs, 'r') as f:
            final_agent_features_list = json.load(f)
        num_agent_features = len(final_agent_features_list)

        # Model parameters
        lookback_window = int(env_cfg.get('lookback_window', 30))
        # Observation: 4 (portfolio state) + 1 (market regime) + lookback * num_agent_features
        calculated_input_dims = 4 + 1 + lookback_window * num_agent_features 
        
        # n_actions: based on position_size_fractions in environment config
        position_size_fractions = env_cfg.get('position_size_fractions', [0.125, 0.25]) # Default if not in env_cfg
        n_actions = 1 + 2 * len(position_size_fractions)

        h1 = int(actor_critic_cfg.get('hidden_size1', 512))
        h2 = int(actor_critic_cfg.get('hidden_size2', 256))
        
        device_str = app_config_loaded.get('device', 'cpu') # Use 'device' from global config
        device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {device}")

        logger.info(f"Initializing ActorCritic model with input_dims={calculated_input_dims}, n_actions={n_actions}, h1={h1}, h2={h2}")
        model_instance = ActorCritic(calculated_input_dims, n_actions, h1=h1, h2=h2)
        
        model_instance.load_state_dict(torch.load(model_path_abs, map_location=device))
        model_instance.to(device)
        model_instance.eval() 
        
        model = model_instance # Assign to global model variable
        model_loaded_successfully = True
        logger.info(f"Model weights loaded successfully from: {model_path_abs} to device '{device}'.")

    except Exception as e:
        model_loaded_successfully = False
        model_loading_error = f"General error during startup: {str(e)}"
        logger.error(f"Error during API startup: {e}", exc_info=True)

# --- API Endpoints ---
@app.get("/health", summary="Health Check", tags=["Status"])
async def health_check():
    """Returns a simple status indicating the API is running."""
    return {"status": "RL Trading Agent Status API is healthy and running"}

@app.get("/model-status", summary="Check Trained Model Loading Status", tags=["Status"])
async def get_model_status():
    """Checks if the trained RL model (specified in config/config.yaml) was successfully loaded on startup."""
    if model_loaded_successfully:
        return {"model_status": "loaded_successfully", 
                "details": "Model .pth file loaded into ActorCritic structure.",
                "model_path_used": app_config_loaded.get('training_loop',{}).get("model_save_directory", "N/A") if app_config_loaded else "Config not loaded"}
    else:
        return {
            "model_status": "load_failed", 
            "error_details": model_loading_error or "Unknown error during model loading."
        }

if __name__ == "__main__":
    logger.info("Starting Uvicorn server programmatically for RL Trading Agent Status API...")
    uvicorn.run(
        "app:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True 
    )