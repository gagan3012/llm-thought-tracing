import numpy as np
import torch
from typing import List, Dict, Optional

def extract_concept_activations(model, prompt: str,
                               intermediate_concepts: List[str],
                               final_concepts: List[str],
                               logit_threshold: float = 0.001) -> Dict:
    """
    Extract evidence of concept activations across all layers and positions.
    
    Parameters:
    -----------
    model : HookedTransformer
        The transformer model to analyze
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
        except:
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