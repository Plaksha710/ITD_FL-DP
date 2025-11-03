# eval_globol.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import os
from model import TabularMLP 
from utils import load_full_dataset
import numpy as np # <-- ADD THIS IMPORT
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GLOBAL_MODEL_PATH_DP = "global_model_dp.pth"
GLOBAL_MODEL_PATH_NON_DP = "global_model.pth" 
CSV_PATH = "data/partition_4.csv"
BATCH_SIZE = 32


# --- NEW FUNCTION TO SAVE PREDICTIONS ---
def save_predictions(labels, preds, filename):
    """Saves true labels and predicted labels to a CSV file."""
    try:
        df = pd.DataFrame({'true_label': labels, 'predicted_label': preds})
        df.to_csv(filename, index=False)
        print(f"✅ Saved predictions to: {filename}")
    except Exception as e:
        print(f"❌ Error saving predictions to {filename}: {e}")
        

def save_final_metrics(results_dict, filename):
    """Saves final evaluation metrics to a two-column CSV (Metric, Value) for visualization."""
    try:
        rounded_results = {k: round(v, 4) for k, v in results_dict.items()}
        df = pd.DataFrame(rounded_results.items(), columns=['Metric', 'Value'])
        df['Metric'] = df['Metric'].str.title()
        df.to_csv(filename, index=False)
        print(f"✅ Saved final evaluation metrics to: {filename}")
    except Exception as e:
        print(f"❌ Error saving final metrics to {filename}: {e}")

# MODIFIED Evaluation function
def evaluate_global(model, data_loader, loss_fn, device):
    """Evaluates the model and returns metrics AND raw labels/predictions.""" # <-- MODIFIED DOCSTRING
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = loss_fn(out, y) 
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

            probs = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    if len(set(all_labels)) < 2:
        auc = 0.0
    else:
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0 

    metrics = {
        "Loss": avg_loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC": auc,
    }
    
    rounded_metrics = {k: round(v, 4) for k, v in metrics.items()}

    # --- RETURN LABELS AND PREDICTIONS ---
    return metrics, rounded_metrics, all_labels, all_preds


# MODIFIED loading and evaluation handler
def load_and_evaluate(model_path, model_name, data_loader, loss_fn, in_dim, device, save_filename, save_preds_filename): # <-- ADDED ARG
    """Handles loading, evaluation, display, and saving for a single model."""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found for {model_name} at: {model_path}. Skipping evaluation.")
        return

    try:
        saved_state_dict = torch.load(model_path, map_location=device)
        print(f"✅ Loaded {model_name} model from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model {model_name} from {model_path}: {e}")
        return

    model = TabularMLP(input_dim=in_dim, output_dim=2)
    model.load_state_dict(saved_state_dict)
    model.to(device)

    # --- UNPACK NEW RETURN VALUES ---
    metrics, rounded_metrics, labels, preds = evaluate_global(model, data_loader, loss_fn, device)

    print("\n" + "="*50)
    print(f"✅ {model_name} Evaluation on Unseen Dataset (Partition 4)")
    print("="*50)
    for k, v in rounded_metrics.items():
        print(f"{k}: {v}")
    print("="*50)

    save_final_metrics(metrics, save_filename)
    # --- SAVE THE PREDICTIONS ---
    save_predictions(labels, preds, save_preds_filename)


if __name__ == "__main__":
    # --- NEW: ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Global Model Evaluation")
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        help="The noise multiplier (sigma) of the DP model to evaluate. If not provided, only the non-DP model is evaluated."
    )
    args = parser.parse_args()

    # --- CORRECTED DATA LOADING ---
    # Import the correct function from utils.py
    from utils import load_full_dataset

    # Call the correct function which returns 2 values
    full_test_dataset, in_dim = load_full_dataset(CSV_PATH)

    # Define UNWEIGHTED Loss for true loss evaluation on the test set
    loss_fn = nn.CrossEntropyLoss()

    # Create the test_loader from the entire dataset
    test_loader = DataLoader(full_test_dataset, batch_size=BATCH_SIZE)

    if len(full_test_dataset) == 0:
        print("❌ Error: Test dataset is empty. Check your data loading.")
        exit(1)

    print(f"✅ Evaluating on the ENTIRE unseen dataset of {len(full_test_dataset)} samples from {CSV_PATH}...")

    # Scenario 2: Evaluate Non-DP Model (always runs)
    load_and_evaluate(
        "global_model.pth", 
        "Non-DP Weights Model (Scenario 2)", 
        test_loader, 
        loss_fn, 
        in_dim, 
        DEVICE, 
        'final_non_dp_eval.csv',
        'predictions_non_dp.csv'
    )

    # Scenario 3: Evaluate DP Model (only runs if noise_multiplier is passed)
    if args.noise_multiplier is not None:
        noise_sigma = args.noise_multiplier
        model_path = f"global_model_dp_sigma_{noise_sigma}.pth"

        load_and_evaluate(
            model_path, 
            f"DP + Weights Model (σ={noise_sigma})", 
            test_loader, 
            loss_fn, 
            in_dim, 
            DEVICE, 
            f'final_dp_eval_sigma_{noise_sigma}.csv',
            f'predictions_dp_sigma_{noise_sigma}.csv'
        )

    print("\nEvaluation complete.")