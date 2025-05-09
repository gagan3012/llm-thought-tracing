"""
LTR - LLM Thought Tracing: Mechanistic Interpretability for Neural Reasoning

A library for tracing and visualizing concept evolution in Large Language Models
"""

from ltr.concept_extraction import extract_concept_activations, get_layer_pattern_and_count
from ltr.reasoning_analysis import analyze_reasoning_paths
from ltr.causal_intervention import perform_causal_intervention
from ltr.visualization import (
    plot_concept_activations,
    plot_causal_intervention_heatmap,
    animate_concept_evolution
)

__version__ = "0.1.0"
