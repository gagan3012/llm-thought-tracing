import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from baukit import TraceDict
import logging


def analyze_attention_patterns(
    model, 
    tokenizer, 
    prompt: str,
    target_heads: Optional[List[Tuple[int, int]]] = None,
    concepts: Optional[List[str]] = None
) -> Dict:
    """
    Analyzes attention patterns in the model for a given prompt.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        prompt: Input text to analyze
        target_heads: List of (layer, head) tuples to analyze. If None, analyzes all heads
        concepts: List of concepts to track in attention patterns
        
    Returns:
        Dict containing attention analysis results
    """
    # Get model architecture information
    model_type = model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    
    # Configure layer and attention patterns based on model type
    if "llama" in model_type or "mistral" in model_type or "qwen" in model_type:
        # For decoder-only models like LLaMA, Mistral, Qwen
        attn_pattern = "model.layers.{}.self_attn.o_proj"
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
    elif "gpt-neox" in model_type or "gpt_neox" in model_type:
        attn_pattern = "gpt_neox.layers.{}.attention.dense"
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
    elif "gpt2" in model_type:
        attn_pattern = "transformer.h.{}.attn.c_proj"
        n_layers = model.config.n_layer
        n_heads = model.config.n_head
    elif "falcon" in model_type:
        attn_pattern = "transformer.h.{}.self_attention.dense"
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
    else:
        # Default pattern as fallback
        attn_pattern = "model.layers.{}.attention.output.dense"
        n_layers = getattr(model.config, "num_hidden_layers", 12)
        n_heads = getattr(model.config, "num_attention_heads", 12)
    
    # Create attention layer patterns for tracing
    attn_patterns = [attn_pattern.format(layer) for layer in range(n_layers)]
    
    # If no target heads are specified, analyze all heads
    if target_heads is None:
        target_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_tokens = inputs.input_ids.shape[1]
    
    # Prepare results structure
    results = {
        "prompt": prompt,
        "tokens": tokenizer.convert_ids_to_tokens(inputs.input_ids[0]),
        "token_strings": tokenizer.tokenize(prompt),
        "attention_maps": {},
        "head_importance": {},
        "concept_attention": {}
    }
    
    # Trace through the model's attention mechanisms
    with TraceDict(model, attn_patterns) as traces:
        _ = model(**inputs)
        
        # Extract attention patterns for each layer
        for layer, layer_pattern in enumerate(attn_patterns):
            if layer_pattern in traces:
                output = traces[layer_pattern]
                
                # For each target head in this layer
                for head in range(n_heads):
                    if (layer, head) not in target_heads:
                        continue
                    
                    # Extract the attention pattern - implementation depends on model architecture
                    attn_pattern = extract_attention_pattern(model, layer, head, inputs, output)
                    
                    if attn_pattern is not None:
                        results["attention_maps"][(layer, head)] = attn_pattern
                        
                        # Calculate attention entropy as a measure of head focus
                        attn_entropy = -np.sum(attn_pattern * np.log(attn_pattern + 1e-10), axis=1)
                        results["head_importance"][(layer, head)] = 1.0 - (attn_entropy / np.log(n_tokens))
    
    # If concepts are provided, analyze attention to these concepts
    if concepts and len(concepts) > 0:
        for concept in concepts:
            # Find token positions for this concept
            concept_positions = find_concept_positions(tokenizer, prompt, concept)
            
            if concept_positions:
                results["concept_attention"][concept] = {}
                
                # For each attention head, calculate average attention to concept tokens
                for (layer, head), attn_map in results["attention_maps"].items():
                    concept_attention = np.mean([np.mean(attn_map[:, pos]) for pos in concept_positions])
                    results["concept_attention"][concept][(layer, head)] = concept_attention
    
    return results


def extract_attention_pattern(model, layer, head, inputs, output):
    """
    Extract attention pattern for a specific head.
    Implementation depends on model architecture.
    """
    # Default extraction method (may need to be adapted for specific models)
    model_type = model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    
    # Different extraction logic based on model architecture
    try:
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # For GPT-2 style models
            return model.transformer.h[layer].attn.attn[0, head].detach().cpu().numpy()
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            # For LLaMA, Mistral, etc.
            return model.model.layers[layer].self_attn.get_attention_scores().detach().cpu().numpy()[0, head]
        else:
            # Fallback method (less accurate)
            n_tokens = inputs.input_ids.shape[1]
            attention = np.eye(n_tokens)  # Default to identity matrix if can't extract
            logging.warning(f"Could not extract attention pattern for layer {layer}, head {head}. Using identity matrix.")
            return attention
    except Exception as e:
        logging.error(f"Error extracting attention pattern: {e}")
        return None


def find_concept_positions(tokenizer, prompt, concept):
    """
    Find token positions corresponding to a concept in the prompt.
    """
    # Tokenize the prompt and concept
    prompt_tokens = tokenizer.tokenize(prompt)
    concept_tokens = tokenizer.tokenize(concept)
    concept_len = len(concept_tokens)
    
    positions = []
    for i in range(len(prompt_tokens) - concept_len + 1):
        if prompt_tokens[i:i+concept_len] == concept_tokens:
            positions.extend(range(i, i+concept_len))
    
    return positions


def ablate_attention_patterns(
    model, 
    tokenizer, 
    prompt: str,
    target_heads: List[Tuple[int, int]],
    ablation_factor: float = 0.0
) -> Dict:
    """
    Ablates specified attention heads to measure their importance.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        prompt: Input text to analyze
        target_heads: List of (layer, head) tuples to ablate
        ablation_factor: Factor to scale attention weights (0.0 = complete ablation)
        
    Returns:
        Dict containing ablation results
    """
    # Get model architecture information
    model_type = model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    
    # Configure layer patterns based on model type
    if "llama" in model_type or "mistral" in model_type or "qwen" in model_type:
        attn_pattern = "model.layers.{}.self_attn.o_proj"
        n_layers = model.config.num_hidden_layers
    elif "gpt-neox" in model_type or "gpt_neox" in model_type:
        attn_pattern = "gpt_neox.layers.{}.attention.dense"
        n_layers = model.config.num_hidden_layers
    elif "gpt2" in model_type:
        attn_pattern = "transformer.h.{}.attn.c_proj"
        n_layers = model.config.n_layer
    elif "falcon" in model_type:
        attn_pattern = "transformer.h.{}.self_attention.dense"
        n_layers = model.config.num_hidden_layers
    else:
        # Default pattern as fallback
        attn_pattern = "model.layers.{}.attention.output.dense"
        n_layers = getattr(model.config, "num_hidden_layers", 12)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs.input_ids[0]
    token_strings = tokenizer.convert_ids_to_tokens(input_tokens)
    
    # Get baseline output
    with torch.no_grad():
        baseline_output = model(**inputs)
    baseline_logits = baseline_output.logits.detach().cpu()
    
    # Create a mapping of layer to heads for targeted ablation
    layer_to_heads = {}
    for layer, head in target_heads:
        if layer not in layer_to_heads:
            layer_to_heads[layer] = []
        layer_to_heads[layer].append(head)
    
    # Define a hook function to apply ablation at specified heads
    def ablation_hook(module, input, output, layer, heads):
        """Hook to ablate specific attention heads."""
        # Implementation depends on model architecture
        try:
            # This implementation assumes output has shape [batch, seq_len, hidden_size]
            # and that we're intercepting the output projection of the attention mechanism
            
            # For complete ablation, we iterate through target heads and zero out their contributions
            if ablation_factor == 0.0:
                for head_idx in heads:
                    # Zero out the head's contribution
                    head_size = output.shape[-1] // module.num_heads if hasattr(module, 'num_heads') else output.shape[-1]
                    start_idx = head_idx * head_size
                    end_idx = (head_idx + 1) * head_size
                    output[:, :, start_idx:end_idx] = 0
            else:
                # Scale by ablation factor
                for head_idx in heads:
                    head_size = output.shape[-1] // module.num_heads if hasattr(module, 'num_heads') else output.shape[-1]
                    start_idx = head_idx * head_size
                    end_idx = (head_idx + 1) * head_size
                    output[:, :, start_idx:end_idx] *= ablation_factor
                    
            return output
            
        except Exception as e:
            logging.error(f"Error in ablation hook: {e}")
            return output

    # Apply hooks for ablation
    hooks = []
    for layer, heads in layer_to_heads.items():
        layer_pattern = attn_pattern.format(layer)
        # Get the module from the model
        module = model
        for name in layer_pattern.split('.'):
            if not hasattr(module, name):
                logging.warning(f"Module {name} not found in {layer_pattern}")
                continue
            module = getattr(module, name)
        
        # Register the hook
        hook = module.register_forward_hook(
            lambda mod, inp, out, layer=layer, heads=heads: ablation_hook(mod, inp, out, layer, heads)
        )
        hooks.append(hook)
    
    # Run the model with ablation
    with torch.no_grad():
        ablated_output = model(**inputs)
    ablated_logits = ablated_output.logits.detach().cpu()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate KL divergence between baseline and ablated outputs
    baseline_probs = torch.softmax(baseline_logits, dim=-1)
    ablated_probs = torch.softmax(ablated_logits, dim=-1)
    kl_div = torch.sum(baseline_probs * (torch.log(baseline_probs + 1e-10) - torch.log(ablated_probs + 1e-10)), dim=-1)
    
    # Get top tokens affected by ablation
    token_changes = []
    n_tokens = len(token_strings)
    
    for pos in range(n_tokens - 1):
        baseline_top_tokens = torch.topk(baseline_logits[0, pos], k=5)
        ablated_top_tokens = torch.topk(ablated_logits[0, pos], k=5)
        
        baseline_top = [(tokenizer.decode([id.item()]).strip(), prob.item()) 
                       for id, prob in zip(baseline_top_tokens.indices, torch.softmax(baseline_top_tokens.values, dim=-1))]
        ablated_top = [(tokenizer.decode([id.item()]).strip(), prob.item()) 
                      for id, prob in zip(ablated_top_tokens.indices, torch.softmax(ablated_top_tokens.values, dim=-1))]
        
        token_changes.append({
            "position": pos,
            "token": token_strings[pos],
            "baseline_top": baseline_top,
            "ablated_top": ablated_top,
            "kl_div": kl_div[0, pos].item()
        })
    
    results = {
        "prompt": prompt,
        "tokens": token_strings,
        "target_heads": target_heads,
        "ablation_factor": ablation_factor,
        "token_changes": token_changes,
        "avg_kl_div": kl_div.mean().item()
    }
    
    return results
