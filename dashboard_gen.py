
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def generate_dashboard(export_dir='.'):
    """
    Generates a comprehensive 3-panel dashboard:
    1. Accuracy (Baseline vs Pruned)
    2. Perplexity (Baseline vs Pruned)
    3. Inference Time (Baseline vs Pruned)
    """
    print("[Dashboard] Generating Overall Metrics Dashboard...")
    
    # Placeholder Data - in real usage, these should be parsed from the JSON reports
    # We will try to load them, else use defaults or placeholders
    
    metrics = {
        'Baseline': {'Acc': 0, 'PPL': 0, 'Time': 0},
        'Pruned':   {'Acc': 0, 'PPL': 0, 'Time': 0}
    }
    
    # 1. Try to load Accuracy from 'accuracy_compare.png' data source or recent logs
    # Since we don't have a structured persistent store for the last run's exact numbers easily accessible 
    # without parsing logs, we will look for 'test_metrics.json' if available or prompt user.
    # tailored for the specific environment context:
    
    # Try parsing the simplified JSON reports if we created them
    # For now, we will create a helper to extract data from the console output or saved files.
    # Let's assume the user runs this AFTER the test script.
    
    # Actually, the best way is to have the test script call this function with the data.
    pass

def plot_dashboard(base_metrics, pruned_metrics, filename='overall_dashboard.png'):
    """
    base_metrics: dict {'acc': float, 'ppl': float, 'time': float}
    pruned_metrics: dict {'acc': float, 'ppl': float, 'time': float}
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Performance Overview: Baseline vs. Pruned', fontsize=20, weight='bold')
    
    # 1. Accuracy
    ax = axes[0]
    x = ['Baseline', 'Pruned']
    y = [base_metrics['acc'], pruned_metrics['acc']]
    bars = ax.bar(x, y, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    ax.set_title('Average Accuracy', fontsize=14)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha='center', fontsize=12)
    
    # 2. Perplexity (Lower is Better)
    ax = axes[1]
    y = [base_metrics['ppl'], pruned_metrics['ppl']]
    bars = ax.bar(x, y, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    ax.set_title('Perplexity (Lower is Better)', fontsize=14)
    ax.set_ylabel('PPL')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{bar.get_height():.2f}", ha='center', fontsize=12)

    # 3. Inference Latency (Lower is Better)
    ax = axes[2]
    y = [base_metrics['time'], pruned_metrics['time']]
    bars = ax.bar(x, y, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    ax.set_title('Inference Latency (ms)', fontsize=14)
    ax.set_ylabel('Time (ms)')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.0f} ms", ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"[Dashboard] Saved {filename}")
