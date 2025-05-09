import numpy as np
import torch
from typing import List, Dict
from ltr.concept_extraction import extract_concept_activations

def analyze_reasoning_paths(model, tokenizer, prompt: str, potential_paths: List[List[str]], concept_threshold: float = 0.2) -> Dict:
    """
    Analyze potential reasoning paths using both layer and position information.
    
    Parameters:
    -----------
    model : PreTrainedModel
        The transformer model to analyze
    tokenizer : PreTrainedTokenizer
        The tokenizer corresponding to the model
    prompt : str
        The input text prompt
    potential_paths : List[List[str]]
        List of possible reasoning paths, where each path is a list of concepts
    concept_threshold : float
        Threshold for concept activation significance
        
    Returns:
    --------
    Dict
        Analysis results including best path and path scores
    """
    # For backward compatibility with existing code that passes a HookedTransformer
    if hasattr(model, 'to_str_tokens') and hasattr(model, 'run_with_cache'):
        return _analyze_reasoning_paths_with_transformerlens(model, prompt, potential_paths, concept_threshold)
    
    all_concepts = set(c for path in potential_paths for c in path)
    results = extract_concept_activations(model, tokenizer, prompt, intermediate_concepts=list(all_concepts), final_concepts=[])

    path_results = {
        "prompt": prompt,
        "potential_paths": potential_paths,
        "path_scores": [],
        "path_details": [],
        "concept_results": results
    }

    for path in potential_paths:
        concepts_found = [concept for concept in path if results["activations"].get(concept)]

        if len(concepts_found) < len(path):
            missing = set(path) - set(concepts_found)
            path_results["path_scores"].append({
                "path": path,
                "score": 0.0,
                "complete": False,
                "missing_concepts": list(missing)
            })
            continue

        concept_peaks = []
        for concept in path:
            if not results["activations"].get(concept):
                continue
                
            peak = max(results["activations"][concept], key=lambda x: x["probability"])
            concept_peaks.append({
                "concept": concept,
                "position": peak["position"],
                "peak_layer": peak["layer"],
                "peak_prob": peak["probability"]
            })
        
        if not concept_peaks:
            continue

        position_order = all(concept_peaks[i]["position"] <= concept_peaks[i+1]["position"] for i in range(len(concept_peaks)-1))
        layer_order = all(concept_peaks[i]["peak_layer"] <= concept_peaks[i+1]["peak_layer"] for i in range(len(concept_peaks)-1))
        in_order = position_order and layer_order

        score = 1.0
        if not in_order:
            score *= 0.5
        avg_prob = sum(peak["peak_prob"] for peak in concept_peaks) / len(concept_peaks)
        score *= min(avg_prob / concept_threshold, 1.0)

        path_results["path_scores"].append({
            "path": path,
            "score": score,
            "complete": True,
            "in_order": in_order,
            "avg_prob": avg_prob
        })

        path_results["path_details"].append({
            "path": path,
            "concept_peaks": concept_peaks
        })
    
    # Sort paths by score
    path_results["path_scores"].sort(key=lambda x: x["score"], reverse=True)
    
    # Determine best path
    if path_results["path_scores"]:
        path_results["best_path"] = path_results["path_scores"][0]["path"]
        path_results["best_path_score"] = path_results["path_scores"][0]["score"]
    else:
        path_results["best_path"] = []
        path_results["best_path_score"] = 0.0
        
    return path_results

def _analyze_reasoning_paths_with_transformerlens(model, prompt, potential_paths, concept_threshold=0.2):
    """Legacy function to maintain backward compatibility with TransformerLens models"""
    all_concepts = set(c for path in potential_paths for c in path)
    results = extract_concept_activations(model, prompt, intermediate_concepts=list(all_concepts), final_concepts=[])

    path_results = {
        "prompt": prompt,
        "potential_paths": potential_paths,
        "path_scores": [],
        "path_details": [],
        "concept_results": results
    }

    for path in potential_paths:
        concepts_found = [concept for concept in path if results["activations"].get(concept)]

        if len(concepts_found) < len(path):
            missing = set(path) - set(concepts_found)
            path_results["path_scores"].append({
                "path": path,
                "score": 0.0,
                "complete": False,
                "missing_concepts": list(missing)
            })
            continue

        concept_peaks = []
        for concept in path:
            peak = max(results["activations"][concept], key=lambda x: x["probability"])
            concept_peaks.append({
                "concept": concept,
                "position": peak["position"],
                "peak_layer": peak["layer"],
                "peak_prob": peak["probability"]
            })

        position_order = all(concept_peaks[i]["position"] <= concept_peaks[i+1]["position"] for i in range(len(concept_peaks)-1))
        layer_order = all(concept_peaks[i]["peak_layer"] <= concept_peaks[i+1]["peak_layer"] for i in range(len(concept_peaks)-1))
        in_order = position_order and layer_order

        score = 1.0
        if not in_order:
            score *= 0.5
        avg_prob = sum(peak["peak_prob"] for peak in concept_peaks) / len(concept_peaks)
        score *= min(avg_prob / concept_threshold, 1.0)

        path_results["path_scores"].append({
            "path": path,
            "score": score,
            "complete": True,
            "in_order": in_order,
            "avg_prob": avg_prob
        })

        path_results["path_details"].append({
            "path": path,
            "concept_peaks": concept_peaks
        })
        
    # Sort paths by score
    path_results["path_scores"].sort(key=lambda x: x["score"], reverse=True)
    
    # Determine best path
    if path_results["path_scores"]:
        path_results["best_path"] = path_results["path_scores"][0]["path"]
        path_results["best_path_score"] = path_results["path_scores"][0]["score"]
    else:
        path_results["best_path"] = []
        path_results["best_path_score"] = 0.0
        
    return path_results
