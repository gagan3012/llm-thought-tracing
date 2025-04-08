import numpy as np
import torch
import gc
from typing import List, Dict, Optional

def perform_causal_intervention(model, prompt: str,
                                concepts: List[str],
                                target_positions: Optional[List[int]] = None,
                                patch_positions: Optional[List[int]] = None) -> Dict:
    """
    Perform causal interventions to analyze concept dependencies.
    
    Parameters:
    -----------
    model : HookedTransformer
        The transformer model to analyze
    prompt : str
        The input text prompt
    concepts : List[str]
        Concepts to trace
    target_positions : Optional[List[int]]
        Token positions to target for intervention
    patch_positions : Optional[List[int]]
        Token positions to patch during intervention
        
    Returns:
    --------
    Dict
        Intervention results including token importance scores
    """
    
    tokens = model.to_str_tokens(prompt)
    n_tokens = len(tokens)
    n_layers = model.cfg.n_layers

    if target_positions is None:
        target_positions = list(range(n_tokens - 1))

    if patch_positions is None:
        patch_positions = list(range(n_tokens))

    results = {
        "prompt": prompt,
        "tokens": tokens,
        "concepts": concepts,
        "intervention_grids": {c: {} for c in concepts},
        "token_importance": {c: [] for c in concepts}
    }

    clean_logits, clean_cache = model.run_with_cache(prompt)

    concept_ids = []
    for concept in concepts:
        try:
            concept_ids.append(model.to_single_token(concept))
        except:
            concept_ids.append(-1)

    final_pos = n_tokens - 1

    clean_probs = {}
    for concept, concept_id in zip(concepts, concept_ids):
        if concept_id != -1:
            clean_probs[concept] = clean_logits[0, final_pos, concept_id].item()
        else:
            clean_probs[concept] = 0.0

    replacements = {
        " Dallas": "Chicago", " plus": " minus", " antagonist": " protagonist"
    }

    del clean_logits
    torch.cuda.empty_cache()
    gc.collect()

    for pos in target_positions:
        if tokens[pos].strip().lower() in [".", ",", "?", "!", ":", ";", "the", "a", "an", "of", "to", "in", "is", "and"]:
            continue

        corrupted_tokens = model.to_tokens(prompt).clone()
        token_to_replace = tokens[pos]
        replacement = replacements.get(token_to_replace, " something")
        replacement_id = model.to_single_token(replacement)
        corrupted_tokens[0, pos] = replacement_id

        corrupt_logits, corrupt_cache = model.run_with_cache(corrupted_tokens)

        corrupt_probs = {}
        for concept, concept_id in zip(concepts, concept_ids):
            if concept_id != -1:
                corrupt_probs[concept] = corrupt_logits[0, final_pos, concept_id].item()
            else:
                corrupt_probs[concept] = 0.0

        del corrupt_logits
        torch.cuda.empty_cache()
        gc.collect()

        for concept in concepts:
            effect = clean_probs[concept] - corrupt_probs[concept]
            results["token_importance"][concept].append({
                "position": pos,
                "token": tokens[pos],
                "corrupt_token": replacement,
                "effect": effect
            })

        for concept, concept_id in zip(concepts, concept_ids):
            if concept_id == -1:
                continue

            grid = np.zeros((n_layers, len(patch_positions)))
            for layer_idx in range(n_layers):
                for patch_idx, patch_pos in enumerate(patch_positions):

                    def patching_hook(activations, hook):
                        activations[0, patch_pos, :] = clean_cache[hook.name][0, patch_pos, :].clone()
                        return activations

                    hook_name = f"blocks.{layer_idx}.hook_resid_post"
                    patched_logits = model.run_with_hooks(
                        corrupted_tokens,
                        fwd_hooks=[(hook_name, patching_hook)]
                    )

                    patched_prob = patched_logits[0, final_pos, concept_id].item()
                    base_effect = corrupt_probs[concept] - clean_probs[concept]
                    recovery = (patched_prob - corrupt_probs[concept]) / abs(base_effect) if abs(base_effect) > 0.01 else 0.0
                    grid[layer_idx, patch_idx] = recovery

                    del patched_logits
                    torch.cuda.empty_cache()
                    gc.collect()

            results["intervention_grids"][concept][pos] = {
                "token": tokens[pos],
                "grid": grid,
                "patch_positions": patch_positions
            }

        del corrupt_cache
        torch.cuda.empty_cache()
        gc.collect()

    # Final sorting
    for concept in concepts:
        results["token_importance"][concept] = sorted(
            results["token_importance"][concept],
            key=lambda x: abs(x["effect"]),
            reverse=True
        )

    return results