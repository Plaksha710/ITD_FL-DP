# visualize_results.py (Corrected Version)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
from sklearn.metrics import confusion_matrix

NOISE_VALUES = [0.25, 0.5, 0.75, 1.0] 

STATIC_BENCHMARK_DATA = {
    "Scenario": ["1. No Balance (Baseline)"],
    "Loss": [0.0175], "Accuracy": [0.9800], "Recall": [0.5000],
    "Precision": [1.0000], "F1-Score": [0.6667], "AUC": [1.0000],
}

#DATA LOADER ---
def load_single_scenario_eval(filepath, scenario_name):
    """Loads a single CSV, standardizes metric names, and pivots the data."""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        
        def standardize_metric_name(name):
            name_str = str(name)
            if name_str.upper() == 'AUC': return 'AUC'
            if name_str.lower() == 'f1-score': return 'F1-Score'
            return name_str.title()
            
        df['Metric'] = df['Metric'].apply(standardize_metric_name)
        
        df_pivot = df.set_index('Metric')['Value'].to_frame().T
        df_pivot['Scenario'] = scenario_name
        return df_pivot
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def load_all_results():
    all_dfs = []
    all_dfs.append(pd.DataFrame(STATIC_BENCHMARK_DATA))
    non_dp_df = load_single_scenario_eval('final_non_dp_eval.csv', '2. Weights (Non-DP)')
    if non_dp_df is not None:
        all_dfs.append(non_dp_df)

    for sigma in NOISE_VALUES:
        filepath = f'final_dp_eval_sigma_{sigma}.csv'
        scenario_name = f'3. DP (σ={sigma})'
        dp_df = load_single_scenario_eval(filepath, scenario_name)
        if dp_df is not None:
            dp_df['sigma'] = sigma
            all_dfs.append(dp_df)
    return pd.concat(all_dfs, ignore_index=True)

def save_markdown_table(df, filename):
    """Saves a dataframe to a markdown file with UTF-8 encoding."""
    try:
        # Open the file with explicit utf-8 encoding to handle special characters
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(df.to_markdown(index=False))
        print(f"✅ Markdown summary saved: {filename}")
    except Exception as e:
        print(f"❌ Error saving Markdown table: {e}")

def plot_tradeoff_curve(df):
    dp_df = df[df['sigma'].notna()].sort_values('sigma')
    if dp_df.empty:
        print("Warning: No DP evaluation data found to plot tradeoff curve.")
        return

    plt.figure(figsize=(8, 6), dpi=300)
    metrics_to_plot = ['F1-Score', 'Precision', 'Recall', 'AUC']
    for metric in metrics_to_plot:
        if metric in dp_df.columns:
            numeric_values = pd.to_numeric(dp_df[metric], errors='coerce')
            plt.plot(dp_df['sigma'], numeric_values, marker='o', linestyle='-', label=metric)

    plt.title('Privacy-Utility Tradeoff Curve', fontweight='bold', pad=15)
    plt.xlabel('Noise Multiplier (σ) - Higher is More Private')
    plt.ylabel('Metric Score')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=False)
    plt.xticks(NOISE_VALUES)
    plt.tight_layout()
    plt.savefig('privacy_utility_tradeoff.png', bbox_inches='tight')
    plt.close()
    print("✅ Saved privacy-utility tradeoff plot: privacy_utility_tradeoff.png")

# --- EXECUTION ---
if __name__ == "__main__":
    print("--- Loading Evaluation Data ---")
    results_df = load_all_results()

    if results_df.empty:
        print("❌ No data loaded. Exiting.")
        sys.exit(1)
        
    print("\n--- Final Evaluation Metrics Table ---")
    display_cols = ['Scenario', 'F1-Score', 'Precision', 'Recall', 'AUC', 'Accuracy']
    display_df = results_df.reindex(columns=display_cols).copy()
    for col in display_cols:
        if col != 'Scenario':
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce').map(lambda x: f'{x:.4f}' if pd.notna(x) else 'nan')
    
    print(display_df.to_markdown(index=False))
    
    save_markdown_table(display_df, 'final_results_summary.md')
    
    plot_tradeoff_curve(results_df)
    print("\n✅ All visualizations completed successfully.")