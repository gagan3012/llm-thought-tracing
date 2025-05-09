import numpy as np
import torch
from typing import List, Dict
from baukit import TraceDict
import logging

def get_layer_pattern_and_count(model):
    """
    Determine the appropriate layer pattern and count for different model architectures.
    """
    model_type = model.config.model_type.lower() if hasattr(model.config, 'model_type') else ""
    
    # Configure layer patterns for popular model types
    if "llama" in model_type:
        layer_pattern = "model.layers.{}.post_attention_layernorm"
        n_layers = model.config.num_hidden_layers
    elif "qwen" in model_type:
        layer_pattern = "model.layers.{}.post_attention_layernorm"
        n_layers = model.config.num_hidden_layers
    else:
        # Default pattern for transformers
        layer_pattern = "model.layers.{}.post_attention_layernorm"
        n_layers = getattr(model.config, "n_layer", 
                          getattr(model.config, "num_hidden_layers", 
                                  getattr(model.config, "n_layers", 12)))
        logging.warning(f"Unknown model type '{model_type}'. Using default layer pattern '{layer_pattern}' and {n_layers} layers.")
    
    return layer_pattern, n_layers

def extract_concept_activations(model, tokenizer, prompt: str,
                               intermediate_concepts: List[str],
                               final_concepts: List[str],
                               logit_threshold: float = 0.001) -> Dict:
    """
    Extract evidence of concept activations across all layers and positions.
    
    Parameters:
    -----------
    model : PreTrainedModel
        The transformer model to analyze
    tokenizer : PreTrainedTokenizer
        The tokenizer corresponding to the model
    prompt : str
        The input text prompt
    intermediate_concepts : List[str]
        Concepts that may appear during reasoning
    final_concepts : List[str]
        Concepts that represent final answers
    logit_threshold : float
        Minimum activation threshold to consider
        
    Returns:
    --------
    Dict
        Detailed information about concept activations
    """
    # For backward compatibility with existing code that passes a HookedTransformer
    if hasattr(model, 'to_str_tokens') and hasattr(model, 'run_with_cache'):
        return _extract_with_transformerlens(model, prompt, intermediate_concepts, final_concepts, logit_threshold)
    
    # Determine layer pattern and count for the model
    layer_pattern, n_layers = get_layer_pattern_and_count(model)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    n_tokens = len(tokens)
    all_concepts = intermediate_concepts + final_concepts

    # Create result structure
    results = {
        "prompt": prompt,
        "tokens": tokens,
        "intermediate_concepts": intermediate_concepts,
        "final_concepts": final_concepts,
        "activations": {concept: [] for concept in all_concepts},
        "activation_grid": {concept: np.zeros((n_layers, n_tokens-1)) for concept in all_concepts} 
    }

    # Get token IDs for all concepts
    concept_token_ids = {}
    for concept in all_concepts:
        try:
            # For better compatibility with different tokenizers
            concept_tokens = tokenizer.encode(" " + concept, add_special_tokens=False)
            if len(concept_tokens) == 1:  # Only use single token concepts
                concept_token_ids[concept] = concept_tokens[0]
            else:
                logging.warning(f"Concept '{concept}' maps to multiple tokens. Using first token.")
                concept_token_ids[concept] = concept_tokens[0]
        except Exception as e:
            logging.warning(f"Failed to encode concept '{concept}': {e}")
            continue

    # Set up traces for all layers
    trace_layers = [layer_pattern.format(i) for i in range(n_layers)]
    
    # Run the model with tracing
    with torch.no_grad():
        with TraceDict(model, trace_layers) as traces:
            _ = model(**inputs)  # Run the model but don't need outputs
        
        # Get the output projection matrix
        if hasattr(model, "lm_head"):
            output_weights = model.lm_head.weight
        elif hasattr(model, "cls"):
            output_weights = model.cls.predictions.decoder.weight
        else:
            raise ValueError("Could not locate output projection matrix. This model architecture may not be supported.")
        
        # Process each layer and position
        for layer in range(n_layers):
            layer_name = layer_pattern.format(layer)
            
            # Skip if the layer wasn't traced
            if layer_name not in traces:
                logging.warning(f"Layer '{layer_name}' wasn't traced. Skipping.")
                continue
            
            # Get the layer output
            layer_output = traces[layer_name].output
            
            # Handle different output shapes (some models may return different tensor dimensions)
            if len(layer_output.shape) == 3:  # [batch, seq_len, hidden_dim]
                layer_output = layer_output[0]  # Take the first batch
            elif len(layer_output.shape) == 4:  # Some models include an extra dimension
                layer_output = layer_output[0, 0]
                
            # Start from position 1 to skip the first token
            for pos in range(1, n_tokens):
                # Handle the case where output shape may be [seq_len, hidden_dim] or [hidden_dim]
                if len(layer_output.shape) == 2:  # [seq_len, hidden_dim]
                    residual = layer_output[pos]
                else:  # Unexpected shape, try with best guess
                    residual = layer_output
                    logging.warning(f"Unexpected layer output shape: {layer_output.shape}. Using best guess.")
                
                # Project to the vocabulary space
                projected_logits = residual @ output_weights.T
                
                # Check activation for each concept
                for concept, concept_id in concept_token_ids.items():
                    concept_score = projected_logits[concept_id].item()
                    
                    # Store in activation grid
                    results["activation_grid"][concept][layer, pos-1] = concept_score
                    
                    # Store activations above threshold
                    if concept_score > logit_threshold:
                        results["activations"][concept].append({
                            "layer": layer,
                            "position": pos-1,
                            "probability": concept_score,
                            "context_token": tokens[pos]
                        })
    
    # Calculate maximum probabilities per layer
    results["layer_max_probs"] = {}
    for concept in all_concepts:
        layer_maxes = np.max(results["activation_grid"].get(concept, np.zeros((n_layers, n_tokens-1))), axis=1)
        results["layer_max_probs"][concept] = layer_maxes

    return results

def _extract_with_transformerlens(model, prompt, intermediate_concepts, final_concepts, logit_threshold):
    """Legacy function to maintain backward compatibility with TransformerLens models"""
    tokens = model.to_str_tokens(prompt)
    n_tokens = len(tokens)
    n_layers = model.cfg.n_layers
    all_concepts = intermediate_concepts + final_concepts

    model.cfg.use_attn_result = True

    logits, cache = model.run_with_cache(prompt)
    
    results = {
        "prompt": prompt,
        "tokens": tokens,
        "intermediate_concepts": intermediate_concepts,
        "final_concepts": final_concepts,
        "activations": {concept: [] for concept in all_concepts},
        "activation_grid": {concept: np.zeros((n_layers, n_tokens-1)) for concept in all_concepts} 
    }
    
    concept_token_ids = {}
    for concept in all_concepts:
        try:
            concept_token_ids[concept] = model.to_single_token(concept)
        except Exception as e:
            logging.warning(f"Failed to convert '{concept}' to a single token in TransformerLens: {e}")
            continue

    for layer in range(n_layers):
        for pos in range(1, n_tokens):  # start from 1, not 0
            residual = cache[f"blocks.{layer}.hook_resid_post"][0, pos, :]

            projected_logits = residual @ model.W_U

            for concept in all_concepts:
                concept_id = concept_token_ids.get(concept)
                if concept_id is None:
                    continue

                concept_score = projected_logits[concept_id].item()

                results["activation_grid"][concept][layer, pos-1] = concept_score

                if concept_score > logit_threshold:
                    results["activations"][concept].append({
                        "layer": layer,
                        "position": pos-1,
                        "probability": concept_score,
                        "context_token": tokens[pos]
                    })

    results["layer_max_probs"] = {}
    for concept in all_concepts:
        layer_maxes = np.max(results["activation_grid"][concept], axis=1)
        results["layer_max_probs"][concept] = layer_maxes

    return results
