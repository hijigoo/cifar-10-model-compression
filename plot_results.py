"""
Plot results and generate figures for the report
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_results(results_file='experiments/results/all_results.json'):
    """Load results from JSON file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return pd.DataFrame(results)


def plot_tradeoff_curve(df, save_path='experiments/results/tradeoff_curve.png'):
    """
    Plot sparsity-accuracy tradeoff curve with 95% CI
    
    Args:
        df: DataFrame with results
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get unique methods (exclude dense)
    methods = df[df['method'] != 'dense']['method'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        
        # Group by sparsity and calculate mean, std, CI
        grouped = method_data.groupby('actual_sparsity')['test_accuracy'].agg(['mean', 'std', 'count'])
        
        sparsities = grouped.index.values
        means = grouped['mean'].values
        stds = grouped['std'].values
        counts = grouped['count'].values
        
        # Calculate 95% CI
        ci = 1.96 * stds / np.sqrt(counts)
        
        # Plot
        ax.plot(sparsities, means, 
               marker=markers[i], 
               label=method.replace('_', ' ').title(),
               linewidth=2.5,
               markersize=8,
               color=colors[i])
        
        # Fill CI
        ax.fill_between(sparsities, 
                        means - ci, 
                        means + ci, 
                        alpha=0.2,
                        color=colors[i])
    
    # Add dense baseline
    dense_data = df[df['method'] == 'dense']
    if len(dense_data) > 0:
        dense_acc = dense_data['test_accuracy'].mean()
        ax.axhline(y=dense_acc, color='black', linestyle='--', 
                  linewidth=2, label='Dense Baseline', alpha=0.7)
    
    ax.set_xlabel('Sparsity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Pruning Methods Comparison: Accuracy vs Sparsity', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Tradeoff curve saved to: {save_path}")
    plt.close()


def generate_efficiency_table(df, save_dir='experiments/results'):
    """
    Generate efficiency comparison table
    
    Args:
        df: DataFrame with results
        save_dir: Directory to save tables
    """
    # Select representative sparsity levels
    sparsity_levels = [0.0, 0.5, 0.9, 0.95]
    
    table_data = []
    
    # Add dense models
    dense_data = df[df['method'] == 'dense']
    if len(dense_data) > 0:
        row = {
            'Model': 'ResNet18',
            'Sparsity': '0.00',
            'Pruning Method': 'N/A (dense)',
            'Top-1 Acc (%)': f"{dense_data['test_accuracy'].mean():.1f}",
            'Params (M)': f"{dense_data['params_millions'].mean():.2f}",
            'Size (MB)': f"{dense_data['model_size_mb'].mean():.2f}",
            'Latency (ms)': f"{dense_data['inference_latency_ms'].mean():.1f}" 
                           if dense_data['inference_latency_ms'].notna().any() else 'N/A'
        }
        table_data.append(row)
    
    # Add pruned models
    for method in df[df['method'] != 'dense']['method'].unique():
        for target_sparsity in sparsity_levels[1:]:  # Skip 0.0
            # Find models close to target sparsity
            subset = df[(df['method'] == method) & 
                       (df['actual_sparsity'].between(target_sparsity - 0.05, 
                                                      target_sparsity + 0.05))]
            
            if len(subset) > 0:
                row = {
                    'Model': 'ResNet18',
                    'Sparsity': f"{subset['actual_sparsity'].mean():.2f}",
                    'Pruning Method': method.replace('_', ' ').title(),
                    'Top-1 Acc (%)': f"{subset['test_accuracy'].mean():.1f}",
                    'Params (M)': f"{subset['nonzero_params_millions'].mean():.2f}",
                    'Size (MB)': f"{subset['model_size_mb'].mean():.2f}",
                    'Latency (ms)': f"{subset['inference_latency_ms'].mean():.1f}"
                                   if subset['inference_latency_ms'].notna().any() else 'N/A'
                }
                table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    
    # Save as Markdown
    markdown_path = os.path.join(save_dir, 'efficiency_table.md')
    with open(markdown_path, 'w') as f:
        f.write("# Efficiency Comparison Table\n\n")
        f.write(table_df.to_markdown(index=False))
    print(f"Markdown table saved to: {markdown_path}")
    
    # Save as LaTeX
    latex_path = os.path.join(save_dir, 'efficiency_table.tex')
    with open(latex_path, 'w') as f:
        f.write(table_df.to_latex(index=False, escape=False))
    print(f"LaTeX table saved to: {latex_path}")
    
    # Save as CSV
    csv_path = os.path.join(save_dir, 'efficiency_table.csv')
    table_df.to_csv(csv_path, index=False)
    print(f"CSV table saved to: {csv_path}")
    
    return table_df


def plot_accuracy_vs_params(df, save_path='experiments/results/accuracy_vs_params.png'):
    """Plot accuracy vs number of parameters"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = df['method'].unique()
    colors = {'dense': 'black', 'magnitude': '#1f77b4', 
              'structured': '#ff7f0e', 'lottery_ticket': '#2ca02c'}
    markers = {'dense': '*', 'magnitude': 'o', 
               'structured': 's', 'lottery_ticket': '^'}
    
    for method in methods:
        method_data = df[df['method'] == method]
        ax.scatter(method_data['nonzero_params_millions'],
                  method_data['test_accuracy'],
                  label=method.replace('_', ' ').title(),
                  s=100,
                  alpha=0.7,
                  color=colors.get(method, 'gray'),
                  marker=markers.get(method, 'o'))
    
    ax.set_xlabel('Number of Parameters (Millions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Efficiency: Accuracy vs Parameters', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy vs params plot saved to: {save_path}")
    plt.close()


def plot_method_comparison(df, save_path='experiments/results/method_comparison.png'):
    """Plot comparison of different methods across sparsity levels"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = df[df['method'] != 'dense']['method'].unique()
    
    metrics = [
        ('actual_sparsity', 'test_accuracy', 'Accuracy (%)'),
        ('actual_sparsity', 'nonzero_params_millions', 'Params (M)'),
        ('actual_sparsity', 'model_size_mb', 'Size (MB)'),
        ('actual_sparsity', 'inference_latency_ms', 'Latency (ms)')
    ]
    
    for idx, (x_col, y_col, y_label) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for method in methods:
            method_data = df[df['method'] == method]
            grouped = method_data.groupby(x_col)[y_col].mean()
            ax.plot(grouped.index, grouped.values, 
                   marker='o', label=method.replace('_', ' ').title(),
                   linewidth=2, markersize=6)
        
        ax.set_xlabel('Sparsity', fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Method Comparison', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Method comparison plot saved to: {save_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Generating plots and tables...")
    print("=" * 80)
    
    # Load results
    results_file = 'experiments/results/all_results.json'
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please run evaluate_all.py first.")
        return
    
    df = load_results(results_file)
    print(f"\nLoaded {len(df)} results")
    print(f"Methods: {df['method'].unique()}")
    print(f"Seeds: {df['seed'].unique()}")
    
    # Create results directory
    os.makedirs('experiments/results', exist_ok=True)
    
    # Generate plots
    print("\n" + "-" * 80)
    print("Generating figures...")
    print("-" * 80)
    
    plot_tradeoff_curve(df)
    plot_accuracy_vs_params(df)
    plot_method_comparison(df)
    
    # Generate tables
    print("\n" + "-" * 80)
    print("Generating tables...")
    print("-" * 80)
    
    table_df = generate_efficiency_table(df)
    
    print("\n" + "=" * 80)
    print("All plots and tables generated successfully!")
    print("=" * 80)
    
    # Print table
    print("\nEfficiency Comparison Table:")
    print(table_df.to_string(index=False))


if __name__ == '__main__':
    main()
