#!/usr/bin/env python3
"""
DiffLUT Experiment Results Visualization Script

This script generates comprehensive visualizations of experiment results including:
1. Heatmaps showing max validation accuracy and max train accuracy for different combinations
2. Training curves for top-k performing experiments
3. Gradient and weight evolution heatmaps across different configurations
4. Training duration analysis

All outputs are saved to a timestamped folder in the outputs directory.
"""

import sys
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Add experiments directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'experiments'))
from utils.db_manager import DBManager, Experiment, MeasurementPoint, Metric

# Set plotting style with better aesthetics
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 100

# Use non-interactive backend
plt.switch_backend('Agg')

# Paths
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / 'results' / 'experiments.db'
GRADIENTS_DIR = BASE_DIR / 'results' / 'gradients'

# Create timestamped output directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = BASE_DIR / 'outputs' / f'visualization_{timestamp}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Database: {DB_PATH}")
print(f"Database exists: {DB_PATH.exists()}")
print(f"Gradients directory: {GRADIENTS_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*60)


def load_experiment_data():
    """Load experiment data from database."""
    print("\n1. Loading Experiment Data...")
    
    # Initialize database manager
    db_manager = DBManager(str(DB_PATH))
    session = db_manager.get_session()
    
    # Query all completed experiments
    experiments = session.query(Experiment).filter(
        Experiment.status.in_(['completed', 'running'])
    ).all()
    
    print(f"Total experiments: {len(experiments)}")
    print(f"Completed: {sum(1 for e in experiments if e.status == 'completed')}")
    print(f"Running: {sum(1 for e in experiments if e.status == 'running')}")
    
    # Convert to DataFrame for easier analysis
    exp_data = []
    for exp in experiments:
        exp_data.append({
            'id': exp.id,
            'name': exp.name,
            'status': exp.status,
            'layer_type': exp.layer_type,
            'node_type': exp.node_type,
            'encoder_name': exp.encoder_name,
            'hidden_size': exp.hidden_size,
            'num_layers': exp.num_layers,
            'n': exp.n,
            'epochs': exp.epochs,
            'lr': exp.learning_rate,
            'optimizer': exp.optimizer,
            'dataset': exp.dataset_name
        })
    
    df_experiments = pd.DataFrame(exp_data)
    print(f"\nUnique layer types: {df_experiments['layer_type'].unique()}")
    print(f"Unique node types: {df_experiments['node_type'].unique()}")
    print(f"Unique encoders: {df_experiments['encoder_name'].unique()}")
    
    return db_manager, session, df_experiments


def load_metrics_data(session, df_experiments):
    """Extract metrics for all experiments."""
    print("\n2. Loading Metrics Data...")
    
    metrics_data = []
    for exp_id in df_experiments['id']:
        # Get all measurement points for this experiment
        mps = session.query(MeasurementPoint).filter(
            MeasurementPoint.experiment_id == exp_id
        ).all()
        
        for mp in mps:
            for metric in mp.metrics:
                metrics_data.append({
                    'experiment_id': exp_id,
                    'epoch': mp.epoch,
                    'phase': mp.phase,
                    'metric_name': metric.metric_name,
                    'metric_value': metric.metric_value,
                    'epoch_time': mp.epoch_time,
                    'total_time': mp.total_time
                })
    
    df_metrics = pd.DataFrame(metrics_data)
    print(f"Total metric records: {len(df_metrics)}")
    print(f"Metric names: {df_metrics['metric_name'].unique()}")
    print(f"Phases: {df_metrics['phase'].unique()}")
    
    # Merge experiment info with metrics
    df_full = df_metrics.merge(df_experiments, left_on='experiment_id', right_on='id')
    print(f"Full DataFrame shape: {df_full.shape}")
    
    return df_full


def create_combination_heatmaps(df, dim1, dim2, metric_name='accuracy', phases=['train', 'val']):
    """
    Create heatmaps showing max metric value for combinations of two dimensions.
    
    Args:
        df: Full dataframe with experiments and metrics
        dim1: First dimension (column name, e.g., 'layer_type')
        dim2: Second dimension (column name, e.g., 'node_type')
        metric_name: Metric to visualize (e.g., 'accuracy', 'loss')
        phases: List of phases to show (e.g., ['train', 'val'])
    """
    n_phases = len(phases)
    fig, axes = plt.subplots(1, n_phases, figsize=(8*n_phases, 6))
    if n_phases == 1:
        axes = [axes]
    
    for ax, phase in zip(axes, phases):
        # Filter for this phase and metric
        df_filtered = df[
            (df['phase'] == phase) & 
            (df['metric_name'] == metric_name)
        ].copy()
        
        if len(df_filtered) == 0:
            ax.text(0.5, 0.5, f'No data for {phase} {metric_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            continue
        
        # Get max metric value for each combination
        pivot_data = df_filtered.groupby([dim1, dim2])['metric_value'].max().reset_index()
        pivot_table = pivot_data.pivot(index=dim2, columns=dim1, values='metric_value')
        
        # Create heatmap with better color scheme
        cmap = sns.color_palette("rocket_r", as_cmap=True) if metric_name == 'accuracy' else 'RdYlGn_r'
        
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap=cmap, 
                   ax=ax, cbar_kws={'label': f'Max {metric_name.capitalize()} (%)'}, 
                   vmin=0, vmax=100 if metric_name == 'accuracy' else None,
                   linewidths=0.5, linecolor='gray', 
                   annot_kws={'size': 10, 'weight': 'bold'})
        
        ax.set_title(f'Max {phase.capitalize()} {metric_name.capitalize()}\n{dim1.replace("_", " ").title()} vs {dim2.replace("_", " ").title()}', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(dim1.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel(dim2.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def generate_heatmaps(df_full):
    """Generate all combination heatmaps."""
    print("\n3. Generating Heatmaps...")
    
    # Layer vs Node
    print("  - Creating Layer vs Node heatmaps...")
    fig = create_combination_heatmaps(df_full, 'layer_type', 'node_type', metric_name='accuracy')
    fig.savefig(OUTPUT_DIR / 'heatmap_layer_vs_node.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Layer vs Encoder
    print("  - Creating Layer vs Encoder heatmaps...")
    fig = create_combination_heatmaps(df_full, 'layer_type', 'encoder_name', metric_name='accuracy')
    fig.savefig(OUTPUT_DIR / 'heatmap_layer_vs_encoder.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Node vs Encoder
    print("  - Creating Node vs Encoder heatmaps...")
    fig = create_combination_heatmaps(df_full, 'node_type', 'encoder_name', metric_name='accuracy')
    fig.savefig(OUTPUT_DIR / 'heatmap_node_vs_encoder.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(df, df_experiments, exp_ids, metric_names=None, save_path=None):
    """
    Plot training curves for specified experiments and metrics.
    
    Args:
        df: Full dataframe with experiments and metrics
        df_experiments: DataFrame with experiment info
        exp_ids: List of experiment IDs to plot
        metric_names: List of metric names to plot (None = all)
        save_path: Optional path to save figure
    """
    # Get unique metrics if not specified
    if metric_names is None:
        metric_names = df[df['experiment_id'].isin(exp_ids)]['metric_name'].unique()
    
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics, 2, figsize=(16, 5*n_metrics))
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    
    # Use a better color palette
    colors = sns.color_palette("husl", len(exp_ids))
    
    for row, metric_name in enumerate(metric_names):
        for col, phase in enumerate(['train', 'val']):
            ax = axes[row, col]
            
            for i, exp_id in enumerate(exp_ids):
                # Get data for this experiment, metric, and phase
                data = df[
                    (df['experiment_id'] == exp_id) &
                    (df['metric_name'] == metric_name) &
                    (df['phase'] == phase)
                ].sort_values('epoch')
                
                if len(data) > 0:
                    exp_info = df_experiments[df_experiments['id'] == exp_id].iloc[0]
                    label = f"{exp_id}: {exp_info['layer_type'][:3]}/{exp_info['node_type'][:8]}/{exp_info['encoder_name'][:6]}"
                    
                    ax.plot(data['epoch'], data['metric_value'], 
                           color=colors[i], alpha=0.8, linewidth=2, label=label, marker='o', markersize=3)
            
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_name.capitalize(), fontsize=11, fontweight='bold')
            ax.set_title(f'{phase.capitalize()} {metric_name.capitalize()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add legend to first plot only
            if row == 0 and col == 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=True, shadow=True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  - Saved to {save_path}")
    
    return fig


def generate_training_curves(df_full, df_experiments):
    """Generate training curves for top-k experiments."""
    print("\n4. Generating Training Curves...")
    
    k = 15
    
    # Get max val accuracy for each experiment
    val_acc = df_full[
        (df_full['phase'] == 'val') & 
        (df_full['metric_name'] == 'accuracy')
    ].groupby('experiment_id')['metric_value'].max().reset_index()
    
    val_acc = val_acc.sort_values('metric_value', ascending=False).head(k)
    top_k_ids = val_acc['experiment_id'].tolist()
    
    print(f"  Top {k} experiments by validation accuracy:")
    for i, (exp_id, acc) in enumerate(zip(val_acc['experiment_id'], val_acc['metric_value']), 1):
        exp_info = df_experiments[df_experiments['id'] == exp_id].iloc[0]
        print(f"    {i}. Exp {exp_id}: {acc:.2f}% - {exp_info['layer_type']}/{exp_info['node_type']}/{exp_info['encoder_name']}")
    
    # Plot training curves
    fig = plot_training_curves(
        df_full, 
        df_experiments,
        top_k_ids, 
        metric_names=['loss', 'accuracy'],
        save_path=OUTPUT_DIR / f'training_curves_top{k}.png'
    )
    plt.close(fig)


def load_histogram_data(histogram_file):
    """Load pre-computed histogram data from pickle file."""
    with open(histogram_file, 'rb') as f:
        return pickle.load(f)


def extract_histogram_matrix(data, data_type='weights', layer_filter=None):
    """
    Extract histogram matrix from loaded data.
    
    Returns:
        hist_matrix: array shape (bins, epochs)
        bin_edges: array shape (bins+1,)
        epochs: list of epoch numbers
        param_name: name of the parameter
    """
    epochs_dict = data.get('epochs', {})
    if not epochs_dict:
        return None, None, None, None
    
    epochs = sorted(epochs_dict.keys())
    hist_key = 'weight_histograms' if data_type == 'weights' else 'gradient_histograms'
    
    # Get first parameter matching layer_filter
    first_epoch_data = epochs_dict[epochs[0]][hist_key]
    if layer_filter:
        param_names = [k for k in first_epoch_data.keys() if layer_filter in k]
    else:
        param_names = list(first_epoch_data.keys())[:1]  # First layer only
    
    if not param_names:
        return None, None, None, None
    
    param_name = param_names[0]
    
    # Get global bin range
    all_edges = [epochs_dict[e][hist_key][param_name]['bin_edges'] 
                 for e in epochs if param_name in epochs_dict[e][hist_key]]
    global_min = min(edges[0] for edges in all_edges)
    global_max = max(edges[-1] for edges in all_edges)
    num_bins = data.get('num_bins', 256)
    global_bins = np.linspace(global_min, global_max, num_bins + 1)
    
    # Build matrix
    hist_matrix = np.zeros((num_bins, len(epochs)), dtype=float)
    
    for i, epoch in enumerate(epochs):
        if param_name in epochs_dict[epoch][hist_key]:
            param_data = epochs_dict[epoch][hist_key][param_name]
            local_hist = param_data['hist']
            local_edges = param_data['bin_edges']
            local_centers = 0.5 * (local_edges[:-1] + local_edges[1:])
            
            for count, center in zip(local_hist, local_centers):
                bin_idx = np.digitize(center, global_bins) - 1
                bin_idx = np.clip(bin_idx, 0, num_bins - 1)
                hist_matrix[bin_idx, i] += count
    
    return hist_matrix, global_bins, epochs, param_name


def plot_gradient_comparison_grid(df_experiments, exp_to_gradient, 
                                 fixed_layer='random', fixed_encoder='thermometer',
                                 data_type='weights'):
    """
    Create a grid of heatmaps comparing gradients/weights across node types and layers.
    Each row = different node type, each column = different layer in the network.
    """
    # Find experiments matching criteria
    matching = df_experiments[
        (df_experiments['layer_type'] == fixed_layer) &
        (df_experiments['encoder_name'] == fixed_encoder) &
        (df_experiments['status'] == 'completed')
    ]
    
    # Group by node type
    node_types = matching['node_type'].unique()
    print(f"    Found {len(node_types)} node types: {node_types}")
    
    # Filter to only those with gradient files
    exp_data = []
    for node_type in node_types:
        exps = matching[matching['node_type'] == node_type]
        for _, exp in exps.iterrows():
            if exp['id'] in exp_to_gradient:
                exp_data.append({
                    'node_type': node_type,
                    'exp_id': exp['id'],
                    'gradient_file': exp_to_gradient[exp['id']],
                    'num_layers': exp['num_layers']
                })
                break  # Take first matching experiment
    
    if not exp_data:
        print("    No experiments found with gradient files!")
        return None
    
    print(f"    Plotting {len(exp_data)} experiments")
    
    # Determine number of layers to plot (use max across experiments)
    max_layers = max(ed['num_layers'] for ed in exp_data)
    print(f"    Max layers to plot: {max_layers}")
    
    # Create grid: rows = node types, columns = layers
    n_rows = len(exp_data)
    n_cols = max_layers
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    
    # Use better colormap
    cmap = sns.color_palette("mako", as_cmap=True)
    
    for row_idx, ed in enumerate(exp_data):
        # Load histogram data once per experiment
        try:
            data = load_histogram_data(ed['gradient_file'])
            
            # Plot each layer
            for col_idx in range(max_layers):
                ax = axes[row_idx, col_idx]
                
                # Get layer name pattern (layers.0, layers.1, etc.)
                layer_filter = f'layers.{col_idx}'
                
                hist_matrix, bin_edges, epochs, param_name = extract_histogram_matrix(
                    data, data_type=data_type, layer_filter=layer_filter
                )
                
                if hist_matrix is None or col_idx >= ed['num_layers']:
                    ax.text(0.5, 0.5, f'No data', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Normalize to density per epoch
                col_sums = hist_matrix.sum(axis=0, keepdims=True)
                col_sums[col_sums == 0] = 1
                hist_matrix = hist_matrix / col_sums
                
                # Plot heatmap with better colormap
                im = ax.imshow(hist_matrix, origin='lower', aspect='auto', cmap=cmap,
                              extent=(min(epochs)-0.5, max(epochs)+0.5, bin_edges[0], bin_edges[-1]))
                
                ax.set_xlabel('Epoch', fontsize=9)
                ax.set_ylabel(f'{data_type.capitalize()}', fontsize=9)
                
                # Title includes node type (for first column) and layer index
                if col_idx == 0:
                    ax.set_title(f'{ed["node_type"]}\nLayer {col_idx}', fontsize=10, fontweight='bold')
                else:
                    ax.set_title(f'Layer {col_idx}', fontsize=10)
                
                # Add colorbar only to the last column
                if col_idx == n_cols - 1:
                    cbar = plt.colorbar(im, ax=ax, label='Density')
                    cbar.ax.tick_params(labelsize=8)
                
        except Exception as e:
            print(f"    Error loading {ed['gradient_file']}: {e}")
            for col_idx in range(max_layers):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, f'Error', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Add overall title
    fig.suptitle(f'{data_type.capitalize()} Distribution Evolution\n{fixed_layer} layer, {fixed_encoder} encoder', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    return fig


def generate_gradient_heatmaps(df_experiments):
    """Generate gradient and weight evolution heatmaps."""
    print("\n5. Generating Gradient/Weight Evolution Heatmaps...")
    
    # Find experiments with gradient snapshots
    gradient_files = list(GRADIENTS_DIR.glob('exp_*_histograms.pkl'))
    print(f"  Found {len(gradient_files)} gradient histogram files")
    
    if len(gradient_files) == 0:
        print("  Skipping gradient heatmaps (no gradient files found)")
        return
    
    # Map experiment IDs to gradient files
    exp_to_gradient = {}
    for gf in gradient_files:
        exp_id = int(gf.stem.split('_')[1])
        exp_to_gradient[exp_id] = gf
    
    # Plot weight evolution comparison
    print("  - Creating weight evolution heatmaps...")
    fig = plot_gradient_comparison_grid(
        df_experiments, 
        exp_to_gradient,
        fixed_layer='random',
        fixed_encoder='thermometer',
        data_type='weights'
    )
    if fig:
        fig.savefig(OUTPUT_DIR / 'weight_evolution_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Plot gradient evolution comparison
    print("  - Creating gradient evolution heatmaps...")
    fig = plot_gradient_comparison_grid(
        df_experiments, 
        exp_to_gradient,
        fixed_layer='random',
        fixed_encoder='thermometer',
        data_type='gradients'
    )
    if fig:
        fig.savefig(OUTPUT_DIR / 'gradient_evolution_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def calculate_duration_data(df_full, df_experiments):
    """Calculate average training duration per epoch for each experiment."""
    print("\n6. Calculating Training Duration Data...")
    
    duration_data = []
    
    for exp_id in df_experiments['id']:
        # Get training phase measurements
        train_mps = df_full[
            (df_full['experiment_id'] == exp_id) & 
            (df_full['phase'] == 'train')
        ][['epoch', 'epoch_time']].drop_duplicates()
        
        if len(train_mps) > 0:
            avg_epoch_time = train_mps['epoch_time'].mean()
            
            # Get max validation accuracy for this experiment
            val_acc = df_full[
                (df_full['experiment_id'] == exp_id) & 
                (df_full['phase'] == 'val') & 
                (df_full['metric_name'] == 'accuracy')
            ]['metric_value'].max()
            
            exp_info = df_experiments[df_experiments['id'] == exp_id].iloc[0]
            
            duration_data.append({
                'experiment_id': exp_id,
                'layer_type': exp_info['layer_type'],
                'node_type': exp_info['node_type'],
                'encoder_name': exp_info['encoder_name'],
                'avg_epoch_time': avg_epoch_time,
                'max_val_accuracy': val_acc if not pd.isna(val_acc) else 0
            })
    
    df_duration = pd.DataFrame(duration_data)
    print(f"  Duration data shape: {df_duration.shape}")
    print(f"  Epoch time range: {df_duration['avg_epoch_time'].min():.2f}s - {df_duration['avg_epoch_time'].max():.2f}s")
    
    return df_duration


def generate_duration_scatter_plots(df_duration):
    """Generate scatter plots of accuracy vs training duration (log-log scale with aggregated means)."""
    print("\n7. Generating Training Duration Scatter Plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Aggregate by taking mean for each category
    # Color by layer type
    layer_agg = df_duration.groupby('layer_type').agg({
        'avg_epoch_time': 'mean',
        'max_val_accuracy': 'mean'
    }).reset_index()
    
    layer_types = layer_agg['layer_type'].unique()
    layer_colors = plt.cm.Set3(np.linspace(0, 1, len(layer_types)))
    layer_color_map = dict(zip(layer_types, layer_colors))
    
    for _, row in layer_agg.iterrows():
        axes[0].scatter(row['avg_epoch_time'], row['max_val_accuracy'], 
                       label=row['layer_type'], alpha=0.8, s=150, 
                       color=layer_color_map[row['layer_type']], 
                       edgecolors='black', linewidth=1.5)
    
    axes[0].set_xlabel('Avg Training Duration per Epoch (s, log scale)', fontsize=10)
    axes[0].set_ylabel('Mean Max Validation Accuracy (%, log scale)', fontsize=10)
    axes[0].set_title('Accuracy vs Training Duration\n(Aggregated by Layer Type)', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8, frameon=True, shadow=True)
    axes[0].grid(True, alpha=0.3, which='both', linestyle='--')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    
    # Color by node type
    node_agg = df_duration.groupby('node_type').agg({
        'avg_epoch_time': 'mean',
        'max_val_accuracy': 'mean'
    }).reset_index()
    
    node_types = node_agg['node_type'].unique()
    node_colors = plt.cm.tab10(np.linspace(0, 1, len(node_types)))
    node_color_map = dict(zip(node_types, node_colors))
    
    for _, row in node_agg.iterrows():
        axes[1].scatter(row['avg_epoch_time'], row['max_val_accuracy'], 
                       label=row['node_type'], alpha=0.8, s=150,
                       color=node_color_map[row['node_type']], 
                       edgecolors='black', linewidth=1.5)
    
    axes[1].set_xlabel('Avg Training Duration per Epoch (s, log scale)', fontsize=10)
    axes[1].set_ylabel('Mean Max Validation Accuracy (%, log scale)', fontsize=10)
    axes[1].set_title('Accuracy vs Training Duration\n(Aggregated by Node Type)', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=8, frameon=True, shadow=True)
    axes[1].grid(True, alpha=0.3, which='both', linestyle='--')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    
    # Color by encoder
    encoder_agg = df_duration.groupby('encoder_name').agg({
        'avg_epoch_time': 'mean',
        'max_val_accuracy': 'mean'
    }).reset_index()
    
    encoder_names = encoder_agg['encoder_name'].unique()
    encoder_colors = plt.cm.Paired(np.linspace(0, 1, len(encoder_names)))
    encoder_color_map = dict(zip(encoder_names, encoder_colors))
    
    for _, row in encoder_agg.iterrows():
        axes[2].scatter(row['avg_epoch_time'], row['max_val_accuracy'], 
                       label=row['encoder_name'], alpha=0.8, s=150,
                       color=encoder_color_map[row['encoder_name']], 
                       edgecolors='black', linewidth=1.5)
    
    axes[2].set_xlabel('Avg Training Duration per Epoch (s, log scale)', fontsize=10)
    axes[2].set_ylabel('Mean Max Validation Accuracy (%, log scale)', fontsize=10)
    axes[2].set_title('Accuracy vs Training Duration\n(Aggregated by Encoder Type)', fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=8, frameon=True, shadow=True)
    axes[2].grid(True, alpha=0.3, which='both', linestyle='--')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'accuracy_vs_duration_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Print aggregated statistics
    print("\n  Aggregated Statistics:")
    print("  " + "="*58)
    print("\n  By Layer Type:")
    for _, row in layer_agg.sort_values('max_val_accuracy', ascending=False).iterrows():
        print(f"    {row['layer_type']:15s}: {row['max_val_accuracy']:5.2f}% accuracy, {row['avg_epoch_time']:6.2f}s/epoch")
    
    print("\n  By Node Type:")
    for _, row in node_agg.sort_values('max_val_accuracy', ascending=False).iterrows():
        print(f"    {row['node_type']:15s}: {row['max_val_accuracy']:5.2f}% accuracy, {row['avg_epoch_time']:6.2f}s/epoch")
    
    print("\n  By Encoder:")
    for _, row in encoder_agg.sort_values('max_val_accuracy', ascending=False).iterrows():
        print(f"    {row['encoder_name']:15s}: {row['max_val_accuracy']:5.2f}% accuracy, {row['avg_epoch_time']:6.2f}s/epoch")
    
    # Print efficiency insights
    print("\n  Efficiency Insights:")
    print("  " + "="*58)
    
    # Find experiments with good accuracy and low training time
    efficient_threshold = df_duration['avg_epoch_time'].quantile(0.5)
    accuracy_threshold = df_duration['max_val_accuracy'].quantile(0.75)
    
    efficient_exps = df_duration[
        (df_duration['avg_epoch_time'] <= efficient_threshold) & 
        (df_duration['max_val_accuracy'] >= accuracy_threshold)
    ].sort_values('max_val_accuracy', ascending=False)
    
    print(f"\n  Most Efficient Experiments (fast training + high accuracy):")
    print(f"  Training time <= {efficient_threshold:.2f}s, Accuracy >= {accuracy_threshold:.2f}%")
    print(f"  Found {len(efficient_exps)} efficient experiments")
    
    # Calculate efficiency score
    df_duration['efficiency_score'] = df_duration['max_val_accuracy'] / df_duration['avg_epoch_time']
    best_efficiency = df_duration.nlargest(5, 'efficiency_score')
    
    print(f"\n  Best Efficiency Score (Accuracy / Training Time):")
    for _, row in best_efficiency.iterrows():
        print(f"    Exp {row['experiment_id']}: {row['efficiency_score']:.2f} - "
              f"{row['layer_type']}/{row['node_type']}/{row['encoder_name']} "
              f"({row['max_val_accuracy']:.2f}% / {row['avg_epoch_time']:.2f}s)")


def generate_duration_bar_plots(df_duration):
    """Generate bar plots of average training duration by configuration."""
    print("\n8. Generating Training Duration Bar Plots...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Group by layer type
    layer_duration = df_duration.groupby('layer_type')['avg_epoch_time'].mean().sort_values(ascending=False)
    axes[0].bar(range(len(layer_duration)), layer_duration.values, color='steelblue', alpha=0.7)
    axes[0].set_xticks(range(len(layer_duration)))
    axes[0].set_xticklabels(layer_duration.index, rotation=45, ha='right')
    axes[0].set_ylabel('Avg Epoch Time (s)')
    axes[0].set_title('Average Training Duration per Epoch by Layer Type')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Group by node type
    node_duration = df_duration.groupby('node_type')['avg_epoch_time'].mean().sort_values(ascending=False)
    axes[1].bar(range(len(node_duration)), node_duration.values, color='coral', alpha=0.7)
    axes[1].set_xticks(range(len(node_duration)))
    axes[1].set_xticklabels(node_duration.index, rotation=45, ha='right')
    axes[1].set_ylabel('Avg Epoch Time (s)')
    axes[1].set_title('Average Training Duration per Epoch by Node Type')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Group by encoder
    encoder_duration = df_duration.groupby('encoder_name')['avg_epoch_time'].mean().sort_values(ascending=False)
    axes[2].bar(range(len(encoder_duration)), encoder_duration.values, color='mediumseagreen', alpha=0.7)
    axes[2].set_xticks(range(len(encoder_duration)))
    axes[2].set_xticklabels(encoder_duration.index, rotation=45, ha='right')
    axes[2].set_ylabel('Avg Epoch Time (s)')
    axes[2].set_title('Average Training Duration per Epoch by Encoder Type')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'training_duration_barplot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n  Training Duration Summary:")
    print(f"    Fastest layer type: {layer_duration.idxmin()} ({layer_duration.min():.2f}s)")
    print(f"    Slowest layer type: {layer_duration.idxmax()} ({layer_duration.max():.2f}s)")
    print(f"    Fastest node type: {node_duration.idxmin()} ({node_duration.min():.2f}s)")
    print(f"    Slowest node type: {node_duration.idxmax()} ({node_duration.max():.2f}s)")
    print(f"    Fastest encoder: {encoder_duration.idxmin()} ({encoder_duration.min():.2f}s)")
    print(f"    Slowest encoder: {encoder_duration.idxmax()} ({encoder_duration.max():.2f}s)")


def print_summary_statistics(df_full):
    """Print summary statistics."""
    print("\n9. Summary Statistics")
    print("="*60)
    
    # Best configurations
    val_acc_summary = df_full[
        (df_full['phase'] == 'val') & 
        (df_full['metric_name'] == 'accuracy')
    ].groupby(['layer_type', 'node_type', 'encoder_name'])['metric_value'].agg(['max', 'mean', 'count'])
    
    print("\nTop 10 configurations by max validation accuracy:")
    print(val_acc_summary.sort_values('max', ascending=False).head(10))
    
    # By component
    print("\nBest by layer type:")
    print(df_full[
        (df_full['phase'] == 'val') & 
        (df_full['metric_name'] == 'accuracy')
    ].groupby('layer_type')['metric_value'].agg(['max', 'mean']))
    
    print("\nBest by node type:")
    print(df_full[
        (df_full['phase'] == 'val') & 
        (df_full['metric_name'] == 'accuracy')
    ].groupby('node_type')['metric_value'].agg(['max', 'mean']))
    
    print("\nBest by encoder:")
    print(df_full[
        (df_full['phase'] == 'val') & 
        (df_full['metric_name'] == 'accuracy')
    ].groupby('encoder_name')['metric_value'].agg(['max', 'mean']))


def main():
    """Main execution function."""
    print("="*60)
    print("DiffLUT Experiment Results Visualization")
    print("="*60)
    
    try:
        # Load data
        db_manager, session, df_experiments = load_experiment_data()
        df_full = load_metrics_data(session, df_experiments)
        
        # Generate visualizations
        generate_heatmaps(df_full)
        generate_training_curves(df_full, df_experiments)
        generate_gradient_heatmaps(df_experiments)
        
        # Duration analysis
        df_duration = calculate_duration_data(df_full, df_experiments)
        generate_duration_scatter_plots(df_duration)
        generate_duration_bar_plots(df_duration)
        
        # Summary statistics
        print_summary_statistics(df_full)
        
        # Close database
        session.close()
        db_manager.close()
        
        print("\n" + "="*60)
        print("COMPLETE!")
        print("="*60)
        print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
        print("\nGenerated files:")
        for img_file in sorted(OUTPUT_DIR.glob('*.png')):
            print(f"  - {img_file.name}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
