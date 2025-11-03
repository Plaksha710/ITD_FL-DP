@echo off
REM run_experiments.bat

echo --- Starting Experiment Workflow ---

REM Define the list of noise multipliers (sigma) to test
SET "NOISE_VALUES=0.25 0.5 0.75 1.0"

REM --- 1. Train the baseline Non-DP model (only once) ---
ECHO.
ECHO [PHASE 1] Training the Non-DP baseline model...
python sim_og.py
ECHO ✅ Non-DP model training complete.

REM --- 2. Train a DP model for each noise value ---
ECHO.
ECHO [PHASE 2] Training DP models for each noise value...
FOR %%S IN (%NOISE_VALUES%) DO (
  ECHO --- Training DP model with sigma = %%S ---
  python sim_og.py --dp --noise_multiplier %%S
)
ECHO ✅ All DP models trained successfully.

REM --- 3. Evaluate all trained models ---
ECHO.
ECHO [PHASE 3] Evaluating all models on the hold-out set...
REM First, evaluate the non-DP model
python evaluate_global.py

REM Next, evaluate each DP model
FOR %%S IN (%NOISE_VALUES%) DO (
  ECHO --- Evaluating DP model with sigma = %%S ---
  python evaluate_global.py --noise_multiplier %%S
)
ECHO All models evaluated successfully.

ECHO.
ECHO --- Experiment Workflow Complete! You can now run visualize_results.py ---