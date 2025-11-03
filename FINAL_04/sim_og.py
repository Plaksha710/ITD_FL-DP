import flwr as fl
from utils import load_client_data
from model import TabularMLP
from client_og import FLClient
from torch.utils.data import DataLoader
import torch
import pandas as pd
from collections import OrderedDict
import argparse

# Import both strategies (assuming they are available in your environment)
from server import LoggingFedAvg 
from server import DpLoggingFedAvg 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Argument Parsing (MODIFIED) ---
parser = argparse.ArgumentParser(description="Federated Learning Simulation")
parser.add_argument(
    "--dp", 
    action="store_true", 
    help="If set, run with Differential Privacy. Otherwise, run Non-DP."
)
# NEW ARGUMENT for noise multiplier
parser.add_argument(
    "--noise_multiplier",
    type=float,
    default=0.5,
    help="The noise multiplier (sigma) for Differential Privacy. Only used if --dp is set."
)
args = parser.parse_args()

# --- 2. Configuration based on Flag (MODIFIED) ---
NUM_CLIENTS = 3
NUM_ROUNDS = 100
LEARNING_RATE = 1e-3
CLIP_NORM = 0.5 # Fixed as requested

if args.dp:
    # SCENARIO 3/4: DP + WEIGHTS
    noise_sigma = args.noise_multiplier
    print(f"--- Running Scenario 3: DP + Weights (Noise Multiplier σ = {noise_sigma}) ---")
    
    # Parameterized filenames
    GLOBAL_MODEL_SAVE_PATH = f'global_model_dp_sigma_{noise_sigma}.pth'
    HISTORY_SAVE_PATH = f'simulation_metrics_dp_sigma_{noise_sigma}.csv'
    
    STRATEGY_CLASS = DpLoggingFedAvg
    STRATEGY_KWARGS = {'clip_norm': CLIP_NORM, 'noise_multiplier': noise_sigma}
else:
    # SCENARIO 2: NON-DP WEIGHTS (Default)
    print("--- Running Scenario 2: Non-DP Weights Only ---")
    GLOBAL_MODEL_SAVE_PATH = 'global_model.pth'
    HISTORY_SAVE_PATH = 'simulation_metrics.csv'
    STRATEGY_CLASS = LoggingFedAvg
    STRATEGY_KWARGS = {}

def get_on_fit_config_fn(lr: float, local_epochs: int):
    """Return a function which configures client training."""
    def fit_config(server_round: int):
        return {"lr": lr, "local_epochs": local_epochs}
    return fit_config

def make_client_fn(cid: str):
    cid_int = int(cid) + 1
    csv_path = f"data/partition_{cid_int}.csv"
    
    # Load data: train_ds, test_ds, in_dim, class_weights
    train_ds, test_ds, in_dim, class_weights = load_client_data(csv_path) 
    
    model = TabularMLP(input_dim=in_dim, output_dim=2)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    return FLClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        class_weights=class_weights
    )

if __name__ == "__main__":
    
    # Instantiate the strategy with ALL parameters (FIXED: removed duplicate instantiation)
    strategy = STRATEGY_CLASS(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=get_on_fit_config_fn(lr=LEARNING_RATE, local_epochs=1),
        **STRATEGY_KWARGS
    )
    
    # Run simulation
    fl.simulation.start_simulation(
        client_fn=make_client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    
    # Save metrics using the configured path
    df = pd.DataFrame(strategy.history)
    df.to_csv(HISTORY_SAVE_PATH, index=False)
    print(f"✅ Metrics saved to {HISTORY_SAVE_PATH}")
    
    # Save the final global model using the configured path
    if strategy.global_weights is not None:
        # Must load one partition to get the feature dimension
        _, _, in_dim, _ = load_client_data("data/partition_1.csv") 
        global_model = TabularMLP(input_dim=in_dim, output_dim=2)
        
        # Compatibility handling for Flower parameter structure
        try:
            from flwr.common import parameters_to_ndarrays
        except ImportError:
            def parameters_to_ndarrays(parameters):
                return parameters.tensors if hasattr(parameters, 'tensors') else parameters
            
        weight_arrays = parameters_to_ndarrays(strategy.global_weights)
        state_dict = OrderedDict()
        for k, v in zip(global_model.state_dict().keys(), weight_arrays):
            if hasattr(v, 'dtype') and v.dtype == 'object':
                v = v.astype('float32')
            state_dict[k] = torch.from_numpy(v)
            
        global_model.load_state_dict(state_dict)
        torch.save(global_model.state_dict(), GLOBAL_MODEL_SAVE_PATH)
        print(f"✅ Global model saved to {GLOBAL_MODEL_SAVE_PATH}")