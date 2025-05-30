import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from baukit import TraceDict
import logging
from functools import partial


class PatchscopeAnalyzer:
    """
    Main class for performing patchscope analysis similar to Racing_Thoughts implementation.
    """

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or model.device
        self.model_type = self._get_model_type()
        self.layer_patterns = self._configure_layer_patterns()

    def _get_model_type(self):
        """Determine model architecture type."""
        return (
            self.model.config.model_type.lower()
            if hasattr(self.model.config, "model_type")
            else ""
        )

    def _configure_layer_patterns(self):
        """Configure layer patterns based on model architecture."""
        if (
            "llama" in self.model_type
            or "mistral" in self.model_type
            or "qwen" in self.model_type
        ):
            return {
                "attention": "model.layers.{}.self_attn",
                "mlp": "model.layers.{}.mlp",
                "residual": "model.layers.{}",
                "n_layers": self.model.config.num_hidden_layers,
            }
        elif "gpt-neox" in self.model_type or "gpt_neox" in self.model_type:
            return {
                "attention": "gpt_neox.layers.{}.attention",
                "mlp": "gpt_neox.layers.{}.mlp",
                "residual": "gpt_neox.layers.{}",
                "n_layers": self.model.config.num_hidden_layers,
            }
        elif "gpt2" in self.model_type:
            return {
                "attention": "transformer.h.{}.attn",
                "mlp": "transformer.h.{}.mlp",
                "residual": "transformer.h.{}",
                "n_layers": self.model.config.n_layer,
            }
        else:
            return {
                "attention": "model.layers.{}.attention",
                "mlp": "model.layers.{}.mlp",
                "residual": "model.layers.{}",
                "n_layers": getattr(self.model.config, "num_hidden_layers", 12),
            }

    def get_layer_names(self, component_type="attention", layers=None):
        """Get layer names for tracing."""
        if layers is None:
            layers = range(self.layer_patterns["n_layers"])

        pattern = self.layer_patterns.get(
            component_type, self.layer_patterns["attention"]
        )
        return [pattern.format(layer) for layer in layers]


def perform_patchscope_analysis(
    model,
    tokenizer,
    prompt: str,
    target_layers: Optional[List[int]] = None,
    explanation_prompts: Optional[List[str]] = None,
    max_tokens: int = 50,
    window_size: int = 5,
    target_entities: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Perform comprehensive patchscope analysis following Racing_Thoughts methodology.

    Args:
        model: The language model to analyze
        tokenizer: Model tokenizer
        prompt: Input prompt for analysis
        target_layers: Specific layers to analyze (None for all)
        explanation_prompts: Prompts for model self-explanation
        max_tokens: Maximum tokens to generate
        window_size: Attention window size
        target_entities: Entities to track through generation
        **kwargs: Additional parameters

    Returns:
        Comprehensive analysis results dictionary
    """

    analyzer = PatchscopeAnalyzer(model, tokenizer)

    # Set default parameters
    if target_layers is None:
        target_layers = list(range(0, analyzer.layer_patterns["n_layers"], 2))

    if explanation_prompts is None:
        explanation_prompts = [
            "What concept is the model processing?",
            "What is the model's confidence in this prediction?",
            "Is this factually correct?",
            "What reasoning led to this output?",
        ]

    if target_entities is None:
        target_entities = []

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]

    results = {
        "prompt": prompt,
        "input_tokens": tokenizer.convert_ids_to_tokens(input_ids),
        "layer_explanations": [],
        "entity_traces": {entity: [] for entity in target_entities},
        "attention_patterns": {},
        "generation_trace": [],
        "intervention_effects": {},
        "metadata": {
            "model_type": analyzer.model_type,
            "n_layers": analyzer.layer_patterns["n_layers"],
            "target_layers": target_layers,
        },
    }

    # Prepare entity tracking
    entity_ids = {}
    for entity in target_entities:
        try:
            entity_tokens = tokenizer(entity, add_special_tokens=False).input_ids
            if entity_tokens:
                entity_ids[entity] = entity_tokens[0]
            else:
                entity_ids[entity] = -1
        except Exception as e:
            logging.warning(f"Could not tokenize entity '{entity}': {e}")
            entity_ids[entity] = -1

    # Generate and analyze token by token
    current_input_ids = input_ids.clone()

    for generation_step in range(max_tokens):
        step_results = {
            "step": generation_step,
            "input_length": len(current_input_ids),
            "layer_activations": {},
            "attention_weights": {},
            "entity_probabilities": {},
            "next_token_info": {},
            "explanations": {},
        }

        # Get layer patterns for tracing
        attention_patterns = analyzer.get_layer_names("attention", target_layers)
        mlp_patterns = analyzer.get_layer_names("mlp", target_layers)
        residual_patterns = analyzer.get_layer_names("residual", target_layers)

        # Comprehensive tracing
        all_patterns = attention_patterns + mlp_patterns + residual_patterns

        try:
            with TraceDict(model, all_patterns) as traces:
                # Forward pass
                outputs = model(current_input_ids.unsqueeze(0))
                logits = outputs.logits[0, -1]

                # Get next token
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.argmax(probs).item()
                next_token = tokenizer.decode([next_token_id])

                step_results["next_token_info"] = {
                    "token_id": next_token_id,
                    "token": next_token,
                    "probability": float(probs[next_token_id]),
                    "top_5_tokens": [
                        {
                            "token": tokenizer.decode([tid]),
                            "token_id": int(tid),
                            "probability": float(probs[tid]),
                        }
                        for tid in torch.topk(probs, 5).indices
                    ],
                }

                # Extract layer activations and attention
                for layer_idx in target_layers:
                    layer_data = {}

                    # Attention weights
                    attn_pattern = attention_patterns[target_layers.index(layer_idx)]
                    if attn_pattern in traces:
                        try:
                            attn_output = traces[attn_pattern].output
                            layer_data["attention"] = extract_attention_weights(
                                attn_output, window_size, analyzer.model_type
                            )
                        except Exception as e:
                            logging.warning(
                                f"Could not extract attention for layer {layer_idx}: {e}"
                            )
                            layer_data["attention"] = []

                    # Residual stream activations
                    residual_pattern = residual_patterns[target_layers.index(layer_idx)]
                    if residual_pattern in traces:
                        try:
                            residual_output = traces[residual_pattern].output
                            if isinstance(residual_output, tuple):
                                residual_output = residual_output[0]

                            # Take last token activation
                            activation = residual_output[0, -1].detach().cpu()
                            layer_data["activation_norm"] = float(
                                torch.norm(activation)
                            )
                            layer_data["activation_mean"] = float(
                                torch.mean(activation)
                            )
                            layer_data["activation_std"] = float(torch.std(activation))

                        except Exception as e:
                            logging.warning(
                                f"Could not extract residual for layer {layer_idx}: {e}"
                            )

                    step_results["layer_activations"][layer_idx] = layer_data

                # Entity probability tracking
                for entity, entity_id in entity_ids.items():
                    if entity_id != -1 and entity_id < len(probs):
                        entity_prob = float(probs[entity_id])
                        step_results["entity_probabilities"][entity] = entity_prob
                        results["entity_traces"][entity].append(
                            {
                                "step": generation_step,
                                "position": len(current_input_ids),
                                "probability": entity_prob,
                            }
                        )

                # Perform patchscope explanations on key layers
                if generation_step % 5 == 0:  # Every 5 steps to avoid overhead
                    explanations = perform_layer_explanations(
                        model,
                        tokenizer,
                        traces,
                        target_layers,
                        explanation_prompts,
                        analyzer,
                    )
                    step_results["explanations"] = explanations

        except Exception as e:
            logging.error(f"Error in generation step {generation_step}: {e}")
            step_results["error"] = str(e)

        results["generation_trace"].append(step_results)

        # Update input for next iteration
        current_input_ids = torch.cat(
            [current_input_ids, torch.tensor([next_token_id], device=model.device)]
        )

        # Stop conditions
        if next_token_id == tokenizer.eos_token_id:
            break

    # Final generation text
    results["generated_text"] = tokenizer.decode(
        current_input_ids, skip_special_tokens=True
    )

    # Post-process analysis
    results = add_summary_statistics(results)

    return results


def extract_attention_weights(
    attn_output, window_size: int, model_type: str
) -> List[float]:
    """Extract and process attention weights from model output."""
    try:
        if isinstance(attn_output, tuple):
            # Many models return (hidden_states, attention_weights)
            if len(attn_output) > 1 and attn_output[1] is not None:
                attn_weights = attn_output[1]
            else:
                return []
        else:
            # For models where attention weights are stored differently
            return []

        if attn_weights is None or len(attn_weights.shape) < 3:
            return []

        # Process attention weights based on shape
        if len(attn_weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
            # Average over heads, take last token attention
            attn = attn_weights[0, :, -1, -window_size:].mean(dim=0)
        elif len(attn_weights.shape) == 3:  # [heads, seq_len, seq_len]
            attn = attn_weights[:, -1, -window_size:].mean(dim=0)
        else:
            attn = attn_weights[-1, -window_size:]

        return attn.detach().cpu().numpy().tolist()

    except Exception as e:
        logging.warning(f"Error processing attention weights: {e}")
        return []


def perform_layer_explanations(
    model, tokenizer, traces, target_layers, explanation_prompts, analyzer
) -> Dict[str, Any]:
    """Perform patchscope self-explanations using traced activations."""
    explanations = {}

    for layer_idx in target_layers:
        layer_explanations = []

        # Get layer activation
        residual_pattern = analyzer.get_layer_names("residual", [layer_idx])[0]

        if residual_pattern in traces:
            try:
                activation = traces[residual_pattern].output
                if isinstance(activation, tuple):
                    activation = activation[0]

                # Use activation for explanation prompting
                for prompt in explanation_prompts:
                    try:
                        # This is a simplified version - in practice, you'd want
                        # to properly patch the activation into the model
                        explanation = generate_explanation_from_activation(
                            model, tokenizer, activation, prompt
                        )
                        layer_explanations.append(
                            {"prompt": prompt, "explanation": explanation}
                        )
                    except Exception as e:
                        logging.warning(
                            f"Could not generate explanation for layer {layer_idx}: {e}"
                        )
                        layer_explanations.append(
                            {"prompt": prompt, "explanation": f"Error: {e}"}
                        )

                explanations[layer_idx] = layer_explanations

            except Exception as e:
                logging.warning(
                    f"Could not process layer {layer_idx} for explanations: {e}"
                )

    return explanations


def generate_explanation_from_activation(
    model, tokenizer, activation, explanation_prompt
):
    """Generate explanation by patching activation (simplified version)."""
    # This is a placeholder - in the full Racing_Thoughts implementation,
    # this would involve sophisticated activation patching

    try:
        # Simple approach: use the activation statistics as context
        if isinstance(activation, tuple):
            activation = activation[0]

        activation_stats = {
            "norm": float(torch.norm(activation[0, -1])),
            "mean": float(torch.mean(activation[0, -1])),
            "max": float(torch.max(activation[0, -1])),
        }

        # Generate a simple explanation based on activation patterns
        if activation_stats["norm"] > 10.0:
            return "High activation - strong signal processing"
        elif activation_stats["norm"] < 1.0:
            return "Low activation - weak signal processing"
        else:
            return "Moderate activation - normal processing"

    except Exception as e:
        return f"Could not generate explanation: {e}"


def analyze_entity_trajectories(
    model, tokenizer, prompt: str, entities: List[str], max_tokens: int = 30
) -> Dict[str, Any]:
    """Analyze entity probability trajectories during generation."""
    return perform_patchscope_analysis(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_entities=entities,
        max_tokens=max_tokens,
        target_layers=None,  # Use all layers
        explanation_prompts=["What entity is being considered?"],
    )


def add_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Add summary statistics to analysis results."""
    summary = {
        "total_generation_steps": len(results["generation_trace"]),
        "entities_tracked": list(results["entity_traces"].keys()),
        "layers_analyzed": results["metadata"]["target_layers"],
        "average_entity_probabilities": {},
        "attention_summary": {},
        "key_insights": [],
    }

    # Calculate average entity probabilities
    for entity, trace in results["entity_traces"].items():
        if trace:
            avg_prob = sum(step["probability"] for step in trace) / len(trace)
            summary["average_entity_probabilities"][entity] = avg_prob

    # Identify key insights
    if summary["average_entity_probabilities"]:
        max_entity = max(
            summary["average_entity_probabilities"].items(), key=lambda x: x[1]
        )
        summary["key_insights"].append(
            f"Highest average entity probability: {max_entity[0]} ({max_entity[1]:.3f})"
        )

    results["summary"] = summary
    return results


# Utility functions for compatibility with existing code
def extract_attention_weights_hook(model, layer_pattern: str):
    """Alternative hook-based approach for attention extraction."""
    attention_weights = {}

    def attention_hook(module, input, output):
        if hasattr(module, "attention_weights"):
            attention_weights["weights"] = module.attention_weights
        elif isinstance(output, tuple) and len(output) > 1:
            attention_weights["weights"] = output[1]

    module = dict(model.named_modules())[layer_pattern]
    hook_handle = module.register_forward_hook(attention_hook)

    return hook_handle, attention_weights


# Main interface function that matches your existing API
def analyze_llm_hallucinations_with_patchscopes(
    model,
    tokenizer,
    prompt: str,
    suspected_hallucination: Optional[str] = None,
    entities_of_interest: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Main interface for hallucination analysis using patchscopes.
    Compatible with the comprehensive analysis function.
    """
    return perform_patchscope_analysis(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_entities=entities_of_interest or [],
        explanation_prompts=[
            "Is this information factually correct?",
            "What reasoning led to this prediction?",
            "What is the model's confidence level?",
            "Are there any inconsistencies in the reasoning?",
        ],
        **kwargs,
    )
