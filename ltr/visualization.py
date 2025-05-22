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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
from typing import List, Dict, Optional
import warnings


def plot_concept_activation_heatmap(
    concept_results: Dict,
    selected_concepts: Optional[List[str]] = None,
    compression_factor: int = 2,
    figsize_per_concept=(14, 4),
) -> plt.Figure:
    """
    Plot a compressed heatmap of concept activations across layers and positions.

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
    all_concepts = (
        concept_results["intermediate_concepts"] + concept_results["final_concepts"]
    )

    if selected_concepts is None:
        selected_concepts = all_concepts
    else:
        selected_concepts = [c for c in selected_concepts if c in all_concepts]

    if not selected_concepts:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No valid concepts to display",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        plt.close(fig)
        return fig

    n_concepts = len(selected_concepts)
    total_height = figsize_per_concept[1] * n_concepts
    fig, axes = plt.subplots(
        n_concepts,
        1,
        figsize=(figsize_per_concept[0], total_height),
        sharex=True,
        squeeze=False,
    )

    tokens = concept_results["tokens"][1:]

    cmap = sns.color_palette("rocket", as_cmap=True)

    vmax = max(
        np.max(concept_results["activation_grid"][concept])
        for concept in selected_concepts
    )

    for i, concept in enumerate(selected_concepts):
        ax = axes[i, 0]
        grid = concept_results["activation_grid"][concept]

        n_layers, n_tokens = grid.shape
        compressed_n_layers = (n_layers + compression_factor - 1) // compression_factor

        compressed_grid = np.zeros((compressed_n_layers, n_tokens))

        for j in range(compressed_n_layers):
            start = j * compression_factor
            end = min((j + 1) * compression_factor, n_layers)
            compressed_grid[j, :] = np.mean(grid[start:end, :], axis=0)

        from scipy.ndimage import gaussian_filter

        compressed_grid = gaussian_filter(compressed_grid, sigma=0.8)

        im = ax.imshow(
            compressed_grid, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, origin="lower"
        )

        concept_type = (
            "Intermediate"
            if concept in concept_results["intermediate_concepts"]
            else "Final"
        )
        ax.set_ylabel(f"{concept}\n({concept_type})", fontsize=12, fontweight="light")

        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)

        bin_labels = [
            f"{i * compression_factor}-{min((i + 1) * compression_factor - 1, n_layers - 1)}"
            for i in range(compressed_n_layers)
        ]
        ax.set_yticks(range(compressed_n_layers))
        ax.set_yticklabels(bin_labels, fontsize=8)

        ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5, alpha=0.3)
        ax.set_xticks(np.arange(-0.5, len(tokens), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, compressed_n_layers, 1), minor=True)

    fig.subplots_adjust(right=0.86)

    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Activation Strength", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Concept Activation Evolution (Compressed View)",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=[0, 0, 0.86, 0.97])
    plt.close(fig)
    return fig


def animate_concept_activation_diagonal(
    concept_results: Dict,
    selected_concepts: Optional[List[str]] = None,
    compression_factor: int = 2,
    figsize=(10, 6),
    interval=100,
) -> HTML:
    """
    Animate concept activation flowing diagonally through network layers.

    Parameters:
    -----------
    concept_results : Dict
        Results from extract_concept_activations
    selected_concepts : Optional[List[str]]
        Specific concepts to visualize
    compression_factor : int
        Factor by which to compress layers for visualization
    figsize : tuple
        Figure size
    interval : int
        Animation interval in milliseconds

    Returns:
    --------
    HTML
        HTML animation for Jupyter display
    """

    all_concepts = (
        concept_results["intermediate_concepts"] + concept_results["final_concepts"]
    )
    if selected_concepts is None:
        selected_concepts = all_concepts
    else:
        selected_concepts = [c for c in selected_concepts if c in all_concepts]

    tokens = concept_results["tokens"][1:]

    concept = selected_concepts[0]

    grid = concept_results["activation_grid"][concept]
    n_layers, n_tokens = grid.shape
    compressed_n_layers = (n_layers + compression_factor - 1) // compression_factor

    compressed_grid = np.zeros((compressed_n_layers, n_tokens))
    for j in range(compressed_n_layers):
        start = j * compression_factor
        end = min((j + 1) * compression_factor, n_layers)
        compressed_grid[j, :] = np.max(grid[start:end, :], axis=0)

    from scipy.ndimage import gaussian_filter

    compressed_grid = gaussian_filter(compressed_grid, sigma=0.8)

    vmax = np.max(compressed_grid)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize)

    cmap = sns.color_palette("rocket_r", as_cmap=True)

    im = ax.imshow(
        np.zeros_like(compressed_grid),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        origin="lower",
    )

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7, color="white")

    bin_labels = [
        f"{i * compression_factor}-{min((i + 1) * compression_factor - 1, n_layers - 1)}"
        for i in range(compressed_n_layers)
    ]
    ax.set_yticks(range(compressed_n_layers))
    ax.set_yticklabels(bin_labels, fontsize=7, color="white")

    ax.set_xlabel("Token Position", fontsize=11, color="white")
    ax.set_ylabel("Layer Bin", fontsize=11, color="white")
    ax.set_title(f"Reasoning Activation: {concept}", fontsize=14, color="white")

    ax.set_xticks(np.arange(-0.5, len(tokens), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, compressed_n_layers, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle=":", linewidth=0.3, alpha=0.2)

    fig.colorbar(im, ax=ax, pad=0.01, shrink=0.7, label="Activation Strength")

    max_step = compressed_n_layers + n_tokens - 2

    def update(step):
        partial = np.zeros_like(compressed_grid)
        for l in range(compressed_n_layers):
            for p in range(n_tokens):
                if (l + p) <= step:
                    partial[l, p] = compressed_grid[l, p]
        im.set_data(partial)
        return [im]

    ani = FuncAnimation(
        fig, update, frames=max_step + 1, interval=interval, blit=True, repeat=True
    )

    plt.close(fig)
    return ani


def animate_reasoning_flow(
    path_results: Dict,
    tokens: List[str],
    model_layers: int,
    figsize=(10, 4),
    interval=700,
    compression_factor=2,
    layer_padding=1,
) -> animation.FuncAnimation:
    """
    Animate the flow of reasoning through the model with light theme.

    Parameters:
    -----------
    path_results : Dict
        Results from analyze_reasoning_paths
    tokens : List[str]
        Tokens from the prompt
    model_layers : int
        Number of layers in the model
    figsize : tuple
        Figure size
    interval : int
        Animation interval in milliseconds
    compression_factor : int
        Factor by which to compress layers for visualization
    layer_padding : int
        Number of layers to pad above and below actual content

    Returns:
    --------
    animation.FuncAnimation
        Matplotlib animation of reasoning flow
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_facecolor("white")

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 9

    if not path_results.get("best_path") or not path_results.get("path_details"):
        ax.text(
            0.5,
            0.5,
            "No valid reasoning path found",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return animation.FuncAnimation(fig, lambda x: None, frames=1), fig

    best_path_details = next(
        (
            d
            for d in path_results["path_details"]
            if d["path"] == path_results["best_path"]
        ),
        None,
    )
    if not best_path_details:
        ax.text(
            0.5,
            0.5,
            "No valid path details found",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return animation.FuncAnimation(fig, lambda x: None, frames=1), fig

    concept_peaks = best_path_details["concept_peaks"]
    concepts = [peak["concept"] for peak in concept_peaks]
    positions = [peak["position"] + 1 for peak in concept_peaks]
    layers = [peak["peak_layer"] for peak in concept_peaks]

    max_position = max(positions) + 2
    compressed_layers = [layer // compression_factor for layer in layers]

    # Find min and max used compressed layers to reduce empty space - same as dark version
    min_compressed_layer = max(0, min(compressed_layers) - layer_padding)
    max_compressed_layer = min(
        (model_layers + compression_factor - 1) // compression_factor,
        max(compressed_layers) + layer_padding + 1,
    )

    # Adjust compressed_layers to be relative to min_compressed_layer for visualization
    visual_layers = [layer - min_compressed_layer for layer in compressed_layers]
    visual_max_layer = max_compressed_layer - min_compressed_layer

    # Add padding to the plot boundaries to accommodate arrows
    x_padding = 0.8
    y_padding = 0.8

    # Set plot limits with padding, using the adjusted layer range
    ax.set_xlim(0.5 - x_padding, len(tokens) + 0.5 + x_padding)
    ax.set_ylim(-0.5 - y_padding, visual_max_layer - 0.5 + y_padding)

    ax.set_xlabel("Token (Prompt)", fontsize=10, labelpad=8)
    ax.set_ylabel("Transformer Layers", fontsize=10, labelpad=8)

    # Simplified title for consistency with dark version
    path_str = " → ".join(path_results["best_path"])
    ax.set_title(
        f"Prompt: {path_results['prompt']} Reasoning Path: {path_str}",
        fontsize=12,
        pad=10,
    )

    ax.set_xticks(range(1, len(tokens) + 1))
    xtick_objs = ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)

    # Create labels only for the range of layers we're actually showing
    ax.set_yticks(range(visual_max_layer))
    layer_labels = [
        f"{(i + min_compressed_layer) * compression_factor}-{min((i + min_compressed_layer + 1) * compression_factor - 1, model_layers - 1)}"
        for i in range(visual_max_layer)
    ]
    ytick_objs = ax.set_yticklabels(layer_labels, fontsize=8)

    ax.grid(True, linestyle="--", alpha=0.3)

    # Use different colors for regular and final concept - consistent with dark version
    bubble_color = "#0d6efd"  # Regular bubble color
    final_bubble_color = "#ff4757"  # Final concept color
    arrow_color = "#00b894"
    text_bg_color = "#f1f2f6"
    final_text_bg_color = "#e2e2e2"  # Slightly darker for final answer

    all_bubbles = []
    all_arrows = []
    all_labels = []
    highlight_ticks_x = []
    highlight_ticks_y = []
    xtick_objs_list = ax.get_xticklabels()
    ytick_objs_list = ax.get_yticklabels()

    # Handle overlapping tokens by applying horizontal offsets
    position_counts = {}
    position_indices = {}

    # Count occurrences of each position
    for pos in positions:
        position_counts[pos] = position_counts.get(pos, 0) + 1

    # Calculate indices for each position
    for i, pos in enumerate(positions):
        position_indices[i] = position_counts.get(pos, 0) > 1

    # Track adjusted positions for arrow connections
    adjusted_positions = []

    def reset_ticks():
        for tick in xtick_objs_list:
            tick.set_color("black")
            tick.set_fontweight("normal")
            tick.set_fontsize(8)
        for tick in ytick_objs_list:
            tick.set_color("black")
            tick.set_fontweight("normal")
            tick.set_fontsize(8)

    def init():
        reset_ticks()
        return []

    def animate(frame_idx):
        artists_to_update = []

        if frame_idx < len(positions):
            # Check if this is the final bubble in the reasoning path
            is_final_output = frame_idx == len(positions) - 1

            # Handle overlapping tokens by applying a horizontal offset
            pos_count = position_counts.get(positions[frame_idx], 0)
            pos_offset = 0

            # Apply horizontal offset if multiple concepts at same position
            if pos_count > 1:
                # Find the count of concepts at this position up to this index
                count_before = sum(
                    1 for i in range(frame_idx) if positions[i] == positions[frame_idx]
                )
                # Calculate offset: distribute points around the token position
                pos_offset = (count_before - (pos_count - 1) / 2) * 0.5

            # Store adjusted position for arrow connections
            adjusted_pos = positions[frame_idx] + pos_offset
            adjusted_positions.append(adjusted_pos)

            # Use different color and slightly larger size for final output
            current_bubble_color = (
                final_bubble_color if is_final_output else bubble_color
            )
            bubble_size = 380 if is_final_output else 300
            edge_color = "darkgoldenrod" if is_final_output else "black"
            edge_width = 1.8 if is_final_output else 1.2

            bubble = ax.scatter(
                adjusted_pos,
                visual_layers[frame_idx],
                s=bubble_size,
                c=current_bubble_color,
                edgecolors=edge_color,
                linewidths=edge_width,
                zorder=5,
            )
            all_bubbles.append(bubble)
            artists_to_update.append(bubble)

            # Use bolder text and slightly different background for final output
            current_text_bg = final_text_bg_color if is_final_output else text_bg_color
            font_weight = "bold" if is_final_output else "normal"
            font_size = 9.5 if is_final_output else 8.5

            label = ax.text(
                adjusted_pos,
                visual_layers[frame_idx] + 0.2,
                concepts[frame_idx],
                fontsize=font_size,
                ha="center",
                va="bottom",
                fontweight=font_weight,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc=current_text_bg,
                    ec=edge_color,
                    alpha=0.9,
                ),
                color="black",
                zorder=6,
            )
            all_labels.append(label)
            artists_to_update.append(label)

            highlight_idx = positions[frame_idx] - 1
            if 0 <= highlight_idx < len(xtick_objs_list):
                xtick_objs_list[highlight_idx].set_color("crimson")
                xtick_objs_list[highlight_idx].set_fontweight("bold")
                xtick_objs_list[highlight_idx].set_fontsize(9)
                highlight_ticks_x.append(xtick_objs_list[highlight_idx])
                artists_to_update.append(xtick_objs_list[highlight_idx])

            layer_idx = visual_layers[frame_idx]
            if 0 <= layer_idx < len(ytick_objs_list):
                ytick_objs_list[layer_idx].set_color("#ff5733")
                ytick_objs_list[layer_idx].set_fontweight("bold")
                ytick_objs_list[layer_idx].set_fontsize(9)
                highlight_ticks_y.append(ytick_objs_list[layer_idx])
                artists_to_update.append(ytick_objs_list[layer_idx])

        elif frame_idx < len(positions) + len(positions) - 1:
            artists_to_update.extend(all_labels)
            artists_to_update.extend(highlight_ticks_x)
            artists_to_update.extend(highlight_ticks_y)

            idx = frame_idx - len(positions)
            if idx + 1 < len(positions):
                # Use adjusted positions for arrow connections
                start_x = adjusted_positions[idx]
                start_y = visual_layers[idx]
                end_x = adjusted_positions[idx + 1]
                end_y = visual_layers[idx + 1]

                # Calculate distance between points to adjust curvature
                distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

                # Dynamically adjust the curvature based on distance
                curvature = min(0.2, 0.8 / distance) if distance > 0 else 0.2

                # Use a shallower arc for horizontal arrows
                if abs(end_y - start_y) < abs(end_x - start_x) * 0.3:
                    curvature *= 0.7

                from matplotlib.patches import FancyArrowPatch

                arrow = FancyArrowPatch(
                    (start_x, start_y),
                    (end_x, end_y),
                    connectionstyle=f"arc3,rad={curvature}",
                    arrowstyle="-|>",
                    mutation_scale=18,
                    color=arrow_color,
                    linewidth=2,
                    zorder=7,
                )
                ax.add_patch(arrow)
                all_arrows.append(arrow)
                artists_to_update.append(arrow)

                artists_to_update.extend(all_arrows[:-1])

        return artists_to_update

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(positions) + len(positions) - 1,
        interval=interval,
        blit=True,
        repeat=False,
    )

    # Use a tighter layout but with some padding to accommodate arrows
    plt.tight_layout(pad=1.2)
    return anim


def animate_reasoning_flow_dark(
    path_results,
    tokens,
    model_layers,
    figsize=(10, 3.5),
    interval=700,
    compression_factor=2,
    layer_padding=1,
):
    """
    Fixed version that properly saves animations with original animation behavior,
    prevents arrows from overflowing the image boundaries, and reduces empty layers.
    Uses a dark theme for better presentation.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 9

    if not path_results.get("best_path") or not path_results.get("path_details"):
        ax.text(
            0.5,
            0.5,
            "No valid reasoning path found",
            ha="center",
            va="center",
            fontsize=14,
            color="white",
        )
        ax.axis("off")
        return animation.FuncAnimation(fig, lambda x: None, frames=1), fig

    best_path_details = next(
        (
            d
            for d in path_results["path_details"]
            if d["path"] == path_results["best_path"]
        ),
        None,
    )
    if not best_path_details:
        ax.text(
            0.5,
            0.5,
            "No valid path details found",
            ha="center",
            va="center",
            fontsize=14,
            color="white",
        )
        ax.axis("off")
        return animation.FuncAnimation(fig, lambda x: None, frames=1), fig

    concept_peaks = best_path_details["concept_peaks"]
    concepts = [peak["concept"] for peak in concept_peaks]
    positions = [peak["position"] + 1 for peak in concept_peaks]
    layers = [peak["peak_layer"] for peak in concept_peaks]

    max_position = max(positions) + 2
    compressed_layers = [layer // compression_factor for layer in layers]

    # Find min and max used compressed layers to reduce empty space
    min_compressed_layer = max(0, min(compressed_layers) - layer_padding)
    max_compressed_layer = min(
        (model_layers + compression_factor - 1) // compression_factor,
        max(compressed_layers) + layer_padding + 1,
    )

    # Adjust compressed_layers to be relative to min_compressed_layer for visualization
    visual_layers = [layer - min_compressed_layer for layer in compressed_layers]
    visual_max_layer = max_compressed_layer - min_compressed_layer

    # Add padding to the plot boundaries to accommodate arrows
    x_padding = 0.8
    y_padding = 0.8

    # Set plot limits with padding, using the adjusted layer range
    ax.set_xlim(0.5 - x_padding, len(tokens) + 0.5 + x_padding)
    ax.set_ylim(-0.5 - y_padding, visual_max_layer - 0.5 + y_padding)

    ax.set_xlabel("Token (Prompt)", fontsize=10, color="white", labelpad=8)
    ax.set_ylabel("Transformer Layers", fontsize=10, color="white", labelpad=8)

    # Use the same simplified title format as light version
    path_str = " → ".join(path_results["best_path"])
    ax.set_title(
        f"Prompt: {path_results['prompt']} Reasoning Path: {path_str}",
        fontsize=12,
        pad=10,
        color="white",
    )

    ax.set_xticks(range(1, len(tokens) + 1))
    xtick_objs = ax.set_xticklabels(
        tokens, rotation=45, ha="right", fontsize=8, color="white"
    )

    # Create labels only for the range of layers we're actually showing
    ax.set_yticks(range(visual_max_layer))
    layer_labels = [
        f"{(i + min_compressed_layer) * compression_factor}-{min((i + min_compressed_layer + 1) * compression_factor - 1, model_layers - 1)}"
        for i in range(visual_max_layer)
    ]
    ytick_objs = ax.set_yticklabels(layer_labels, fontsize=8, color="white")

    ax.grid(True, linestyle="--", alpha=0.3, color="gray")

    # Use different colors for regular and final concept - consistent with light version
    bubble_color = "#0d6efd"  # Regular bubble color
    final_bubble_color = "#ff4757"  # Final concept color
    arrow_color = "#00ff99"
    text_bg_color = "#222222"
    final_text_bg_color = "#2a2a2a"  # Slightly darker for final answer

    all_bubbles = []
    all_arrows = []
    all_labels = []
    highlight_ticks_x = []
    highlight_ticks_y = []
    xtick_objs_list = ax.get_xticklabels()
    ytick_objs_list = ax.get_yticklabels()

    # Handle overlapping tokens by applying horizontal offsets
    position_counts = {}
    position_indices = {}

    # Count occurrences of each position
    for pos in positions:
        position_counts[pos] = position_counts.get(pos, 0) + 1

    # Calculate indices for each position
    for i, pos in enumerate(positions):
        position_indices[i] = position_counts.get(pos, 0) > 1

    # Track adjusted positions for arrow connections
    adjusted_positions = []

    def reset_ticks():
        for tick in xtick_objs_list:
            tick.set_color("white")
            tick.set_fontweight("normal")
            tick.set_fontsize(8)
        for tick in ytick_objs_list:
            tick.set_color("white")
            tick.set_fontweight("normal")
            tick.set_fontsize(8)

    def init():
        reset_ticks()
        return []

    def animate(frame_idx):
        artists_to_update = []

        if frame_idx < len(positions):
            # Check if this is the final bubble in the reasoning path
            is_final_output = frame_idx == len(positions) - 1

            # Handle overlapping tokens by applying a horizontal offset
            pos_count = position_counts.get(positions[frame_idx], 0)
            pos_offset = 0

            # Apply horizontal offset if multiple concepts at same position
            if pos_count > 1:
                # Find the count of concepts at this position up to this index
                count_before = sum(
                    1 for i in range(frame_idx) if positions[i] == positions[frame_idx]
                )
                # Calculate offset: distribute points around the token position
                pos_offset = (count_before - (pos_count - 1) / 2) * 0.3

            # Store adjusted position for arrow connections
            adjusted_pos = positions[frame_idx] + pos_offset
            adjusted_positions.append(adjusted_pos)

            # Use different color and slightly larger size for final output
            current_bubble_color = (
                final_bubble_color if is_final_output else bubble_color
            )
            bubble_size = 380 if is_final_output else 300
            edge_color = "gold" if is_final_output else "white"
            edge_width = 1.8 if is_final_output else 1.2

            # Use visual_layers instead of compressed_layers for more compact display
            bubble = ax.scatter(
                adjusted_pos,
                visual_layers[frame_idx],
                s=bubble_size,
                c=current_bubble_color,
                edgecolors=edge_color,
                linewidths=edge_width,
                zorder=5,
            )
            all_bubbles.append(bubble)
            artists_to_update.append(bubble)

            # Use bolder text and slightly different background for final output
            current_text_bg = final_text_bg_color if is_final_output else text_bg_color
            font_weight = "bold" if is_final_output else "normal"
            font_size = 9.5 if is_final_output else 8.5

            # Use visual_layers for text placement too
            label = ax.text(
                adjusted_pos,
                visual_layers[frame_idx] + 0.2,
                concepts[frame_idx],
                fontsize=font_size,
                ha="center",
                va="bottom",
                fontweight=font_weight,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc=current_text_bg,
                    ec=edge_color,
                    alpha=0.9,
                ),
                color="white",
                zorder=6,
            )
            all_labels.append(label)
            artists_to_update.append(label)

            highlight_idx = positions[frame_idx] - 1
            if 0 <= highlight_idx < len(xtick_objs_list):
                xtick_objs_list[highlight_idx].set_color("#ff5e57")
                xtick_objs_list[highlight_idx].set_fontweight("bold")
                xtick_objs_list[highlight_idx].set_fontsize(9)
                highlight_ticks_x.append(xtick_objs_list[highlight_idx])
                artists_to_update.append(xtick_objs_list[highlight_idx])

            # Use the visual layer index for highlighting y ticks
            layer_idx = visual_layers[frame_idx]
            if 0 <= layer_idx < len(ytick_objs_list):
                ytick_objs_list[layer_idx].set_color("#00ffe4")
                ytick_objs_list[layer_idx].set_fontweight("bold")
                ytick_objs_list[layer_idx].set_fontsize(9)
                highlight_ticks_y.append(ytick_objs_list[layer_idx])
                artists_to_update.append(ytick_objs_list[layer_idx])

        elif frame_idx < len(positions) + len(positions) - 1:
            artists_to_update.extend(all_labels)
            artists_to_update.extend(highlight_ticks_x)
            artists_to_update.extend(highlight_ticks_y)

            idx = frame_idx - len(positions)
            if idx + 1 < len(positions):
                start_x = adjusted_positions[idx]
                start_y = visual_layers[idx]
                end_x = adjusted_positions[idx + 1]
                end_y = visual_layers[idx + 1]

                # Calculate distance between points to adjust curvature
                distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

                # Dynamically adjust the curvature based on distance
                curvature = min(0.2, 0.8 / distance) if distance > 0 else 0.2

                # Use a shallower arc for horizontal arrows
                if abs(end_y - start_y) < abs(end_x - start_x) * 0.3:
                    curvature *= 0.7

                from matplotlib.patches import FancyArrowPatch

                arrow = FancyArrowPatch(
                    (start_x, start_y),
                    (end_x, end_y),
                    connectionstyle=f"arc3,rad={curvature}",
                    arrowstyle="-|>",
                    mutation_scale=18,
                    color=arrow_color,
                    linewidth=2,
                    zorder=7,
                )
                ax.add_patch(arrow)
                all_arrows.append(arrow)
                artists_to_update.append(arrow)

                artists_to_update.extend(all_arrows[:-1])

        return artists_to_update

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(positions) + len(positions) - 1,
        interval=interval,
        blit=True,
        repeat=False,
    )

    # Use a tighter layout but with some padding to accommodate arrows
    plt.tight_layout(pad=1.2)
    return anim

def plot_layer_position_intervention(
    intervention_results: Dict,
    selected_concepts: Optional[List[str]] = None,
    top_k_positions: int = 3,
    figsize=(15, 12),
) -> plt.Figure:
    """
    Visualize the effects of causal interventions across layers and positions.

    Parameters:
    -----------
    intervention_results : Dict
        Results from perform_causal_intervention
    selected_concepts : Optional[List[str]]
        Specific concepts to visualize
    top_k_positions : int
        Number of top positions to display
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Matplotlib figure with intervention heatmaps
    """

    replacements = {
        " Dallas": "Chicago",
        " sum": "difference",
        " antagonist": " protagonist",
    }

    all_concepts = intervention_results["concepts"]
    tokens = intervention_results["tokens"]

    if selected_concepts is None:
        selected_concepts = all_concepts
    else:
        selected_concepts = [c for c in selected_concepts if c in all_concepts]

    if not selected_concepts:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No valid concepts to display",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return fig

    top_positions = {
        concept: [
            item["position"]
            for item in intervention_results["token_importance"].get(concept, [])[
                :top_k_positions
            ]
        ]
        for concept in selected_concepts
    }
    max_positions = max((len(pos) for pos in top_positions.values()), default=0)
    if max_positions == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No valid positions to display",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return fig

    n_concepts = len(selected_concepts)
    fig, axes = plt.subplots(
        n_concepts,
        max_positions,
        figsize=(figsize[0], figsize[1] * n_concepts / 2),
        squeeze=False,
    )
    axes = np.atleast_2d(axes)

    cmap = sns.diverging_palette(0, 240, s=100, l=60, as_cmap=True)
    vmin, vmax = 0, 0
    for concept in selected_concepts:
        for pos_data in (
            intervention_results["intervention_grids"].get(concept, {}).values()
        ):
            grid = pos_data["grid"]
            vmin = min(vmin, np.min(grid))
            vmax = max(vmax, np.max(grid))
    limit = max(abs(vmin), abs(vmax))
    vmin, vmax = -limit, limit

    for i, concept in enumerate(selected_concepts):
        concept_grid_data = intervention_results["intervention_grids"].get(concept, {})
        concept_positions = top_positions.get(concept, [])

        for j, pos in enumerate(concept_positions):
            ax = axes[i, j]
            pos_data = concept_grid_data.get(pos, None)

            if pos_data is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.axis("off")
                continue

            grid = pos_data["grid"]
            patch_positions = pos_data["patch_positions"]
            corrupt_token = pos_data["token"]

            im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.invert_yaxis()
            ax.set_xlabel("Patch Position")
            ax.set_ylabel("Layer")
            ax.set_xticks(range(len(patch_positions)))
            ax.set_xticklabels(
                [tokens[p] if p < len(tokens) else "N/A" for p in patch_positions],
                fontsize=8,
                rotation=45,
            )
            ax.set_yticks(range(grid.shape[0]))
            ax.set_yticklabels(range(grid.shape[0]), fontsize=8)
            ax.set_title(f'Corrupting "{corrupt_token}" (pos {pos})', fontsize=9)
            ax.grid(
                which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3
            )
            ax.set_xticks(np.arange(-0.5, len(patch_positions), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)

            if j == 0:
                cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
                cbar.set_label("Recovery Effect", fontsize=10)

    plt.figtext(
        0.5,
        0.01,
        "Each heatmap shows how patching a layer (y-axis) and token position (x-axis) affects recovery of the concept prediction.\n"
        "Red = harmful, Blue = helpful. Values near 1.0 mean strong recovery toward the clean prediction.",
        ha="center",
        fontsize=9,
        bbox={"facecolor": "lightyellow", "alpha": 0.5, "pad": 5},
    )

    fig.suptitle("Causal Tracing: Layer × Position Recovery Maps", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.close(fig)
    return fig


def save_animation(
    path_results, tokens, model_layers, output_path, format="gif", fps=10, dpi=150
):
    """
    Properly save the animation as either GIF or MP4

    Parameters:
    -----------
    path_results: Dict
        The results dictionary
    tokens: List[str]
        List of tokens to display
    model_layers: int
        Number of model layers
    output_path: str
        Path to save the animation (should end with .gif or .mp4)
    format: str
        Either "gif" or "mp4"
    fps: int
        Frames per second
    dpi: int
        DPI for the saved animation
    """
    anim, fig = animate_reasoning_flow_dark(
        path_results=path_results,
        tokens=tokens,
        model_layers=model_layers,
        figsize=(10, 4),
        interval=100,
        compression_factor=2,
    )

    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if format.lower() == "gif":
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
    else:
        try:
            from matplotlib.animation import FFMpegWriter

            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(
                output_path,
                writer=writer,
                dpi=dpi,
                savefig_kwargs={"facecolor": "black"},
            )
        except Exception as e:
            print(f"Failed to use FFMpegWriter: {e}")
            print("Falling back to default writer...")
            anim.save(
                output_path, fps=fps, dpi=dpi, savefig_kwargs={"facecolor": "black"}
            )

    print(f"Animation saved to {output_path}")
    plt.close(fig)

    return anim