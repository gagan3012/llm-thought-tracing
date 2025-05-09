import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional
import warnings

def plot_concept_activations(concept_results: Dict,
                            selected_concepts: Optional[List[str]] = None,
                            compression_factor: int = 2,
                            figsize_per_concept=(14, 4)) -> plt.Figure:
    """
    Plot a heatmap of concept activations across layers and positions.
    
    Parameters:
    -----------
    concept_results : Dict
        Results from extract_concept_activations
    selected_concepts : Optional[List[str]]
        Specific concepts to visualize
    compression_factor : int
        Factor by which to compress layers for visualization
    figsize_per_concept : tuple
        Figure size per concept
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with activation heatmaps
    """
    all_concepts = concept_results["intermediate_concepts"] + concept_results["final_concepts"]

    if selected_concepts is None:
        selected_concepts = all_concepts
    else:
        selected_concepts = [c for c in selected_concepts if c in all_concepts]

    if not selected_concepts:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No valid concepts to display", ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.close(fig)
        return fig

    n_concepts = len(selected_concepts)
    total_height = figsize_per_concept[1] * n_concepts
    fig, axes = plt.subplots(n_concepts, 1, figsize=(figsize_per_concept[0], total_height),
                              sharex=True, squeeze=False)

    tokens = concept_results["tokens"][1:]  # Skip the first token (usually BOS)

    cmap = sns.color_palette("rocket", as_cmap=True)

    vmax = max(np.max(concept_results["activation_grid"][concept]) for concept in selected_concepts)

    for i, concept in enumerate(selected_concepts):
        ax = axes[i, 0]
        grid = concept_results["activation_grid"][concept]

        n_layers, n_tokens = grid.shape
        compressed_n_layers = (n_layers + compression_factor - 1) // compression_factor

        compressed_grid = np.zeros((compressed_n_layers, n_tokens))

        for j in range(compressed_n_layers):
            start = j * compression_factor
            end = min((j+1) * compression_factor, n_layers)
            compressed_grid[j, :] = np.mean(grid[start:end, :], axis=0)

        try:
            from scipy.ndimage import gaussian_filter
            compressed_grid = gaussian_filter(compressed_grid, sigma=0.8)
        except ImportError:
            pass  # Skip smoothing if scipy is not available

        im = ax.imshow(compressed_grid, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, origin='lower')

        concept_type = "Intermediate" if concept in concept_results["intermediate_concepts"] else "Final"
        ax.set_ylabel(f"{concept}\n({concept_type})", fontsize=12, fontweight='light')

        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)

        bin_labels = [f"{i*compression_factor}-{min((i+1)*compression_factor-1, n_layers-1)}"
                      for i in range(compressed_n_layers)]
        ax.set_yticks(range(compressed_n_layers))
        ax.set_yticklabels(bin_labels, fontsize=8)

        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.set_xticks(np.arange(-0.5, len(tokens), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, compressed_n_layers, 1), minor=True)

    fig.subplots_adjust(right=0.86)

    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Activation Strength', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle('Concept Activation Evolution', fontsize=16, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])
    
    return fig

def plot_causal_intervention_heatmap(intervention_results: Dict, 
                                    concept: str,
                                    target_pos: Optional[int] = None,
                                    figsize=(12, 10)) -> plt.Figure:
    """
    Plot a heatmap visualization of causal intervention results.
    
    Parameters:
    -----------
    intervention_results : Dict
        Results from perform_causal_intervention
    concept : str
        The concept to visualize
    target_pos : Optional[int]
        Specific target position to visualize
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with intervention heatmap
    """
    if concept not in intervention_results["concepts"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Concept '{concept}' not found in intervention results", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    intervention_grids = intervention_results["intervention_grids"][concept]
    
    if not intervention_grids:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No intervention grids available", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    if target_pos is None:
        # Find the position with the largest impact
        positions = list(intervention_grids.keys())
        if not positions:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No target positions available", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        token_importance = intervention_results["token_importance"][concept]
        if token_importance:
            # Use the position with the highest impact
            target_pos = max(token_importance, key=lambda x: abs(x["impact"]))["position"]
        else:
            # Fall back to the first position
            target_pos = positions[0]
    
    if target_pos not in intervention_grids:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Target position {target_pos} not found in intervention grids", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    grid = intervention_grids[target_pos]
    tokens = intervention_results["tokens"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    vmax = np.max(np.abs(grid))
    vmin = -vmax
    
    # Create a symmetrical diverging colormap centered at zero
    cmap = sns.color_palette("vlag", as_cmap=True)
    
    im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Add target position marker
    ax.axvline(x=target_pos, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Set up labels and ticks
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    
    # Customize xticks
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=10)
    
    # Customize yticks
    n_layers = grid.shape[0]
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(range(n_layers), fontsize=10)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Effect of Patching (Change in Probability)', fontsize=11)
    
    # Title
    token_to_replace = tokens[target_pos]
    fig.suptitle(f'Causal Intervention: Effect of Patching Each Position After Corrupting "{token_to_replace}"', 
                 fontsize=14, fontweight='bold')
    
    # Subtitle explaining the color scale
    ax.set_title("Blue = Positive Effect (Recovers Original Prediction), Red = Negative Effect", 
                fontsize=10, pad=10)
    
    # Add grid lines for better readability
    ax.grid(False)
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    
    return fig

def animate_concept_evolution(concept_results: Dict, concept: str, figsize=(12, 5)):
    """
    Placeholder function for creating animations of concept evolution.
    In a full implementation, this would animate the activation patterns
    over time to show how concepts evolve through the network.
    
    Parameters:
    -----------
    concept_results : Dict
        Results from extract_concept_activations
    concept : str
        The concept to animate
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    str
        A message indicating that animation functionality requires additional dependencies
    """
    return "Animations require matplotlib animation capabilities and are provided in the full package."
