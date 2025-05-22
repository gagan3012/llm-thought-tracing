import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from baukit import TraceDict
import logging


def perform_patchscope_analysis(
    model, 
    tokenizer, 
    prompt: str,
    target_entities: List[str],
    window_size: int = 3,
    max_tokens: int = 50
) -> Dict:
    """
    Performs patchscope analysis for open-ended generation.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        prompt: Input prompt to analyze
        target_entities: List of entities to track
        window_size: Window size for attention aggregation
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Dict containing patchscope analysis results
    """
    # Get model architecture information
    model_type = model.config.model_type.lower() if hasattr(model.config, "model_type") else ""
    
    # Configure layer patterns based on model type
    if "llama" in model_type or "mistral" in model_type or "qwen" in model_type:
        # For decoder-only models like LLaMA, Mistral, Qwen
        attn_pattern = "model.layers.{}.self_attn"
        n_layers = model.config.num_hidden_layers
    elif "gpt-neox" in model_type or "gpt_neox" in model_type:
        attn_pattern = "gpt_neox.layers.{}.attention"
        n_layers = model.config.num_hidden_layers
    elif "gpt2" in model_type:
        attn_pattern = "transformer.h.{}.attn"
        n_layers = model.config.n_layer
    else:
        # Default pattern as fallback
        attn_pattern = "model.layers.{}.attention"
        n_layers = getattr(model.config, "num_hidden_layers", 12)
    
    # Create attention layer patterns for tracing
    attn_patterns = [attn_pattern.format(layer) for layer in range(n_layers)]
    
    # Initial tokenization
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    n_initial_tokens = len(tokens)
    
    # Prepare results structure
    results = {
        "prompt": prompt,
        "full_generation": None,  # Will be filled at the end
        "entity_traces": {entity: [] for entity in target_entities},
        "token_generations": []
    }
    
    # Full generation tokens
    full_generation_tokens = tokens.copy()
    
    # Prepare entity token IDs
    entity_ids = {}
    for entity in target_entities:
        try:
            # Try to tokenize the entity
            entity_token_ids = tokenizer(entity, add_special_tokens=False).input_ids
            if entity_token_ids:
                entity_ids[entity] = entity_token_ids[0]  # Just take the first token if multi-token
            else:
                logging.warning(f"Entity '{entity}' tokenized to empty list")
                entity_ids[entity] = -1
        except Exception as e:
            logging.error(f"Error tokenizing entity '{entity}': {e}")
            entity_ids[entity] = -1
    
    # Generate tokens one by one
    current_input_ids = input_ids.clone()
    
    for i in range(max_tokens):
        # Create patterns to trace for this generation step
        trace_patterns = attn_patterns
        
        # Trace through the model
        with TraceDict(model, trace_patterns) as traces:
            outputs = model(current_input_ids.unsqueeze(0))
            
            # Get next token prediction
            logits = outputs.logits[0, -1]
            next_token_id = torch.argmax(logits).item()
            next_token = tokenizer.decode([next_token_id]).strip()
            
            # Add to generation
            full_generation_tokens.append(tokenizer.convert_ids_to_tokens([next_token_id])[0])
            
            # Calculate entity probabilities
            token_data = {
                "position": n_initial_tokens + i,
                "token": next_token,
                "token_id": next_token_id,
                "entity_probs": {}
            }
            
            # Get probabilities for each entity
            probs = torch.softmax(logits, dim=-1)
            for entity, entity_id in entity_ids.items():
                if entity_id != -1:
                    entity_prob = probs[entity_id].item()
                    token_data["entity_probs"][entity] = entity_prob
                    
                    # Add to entity trace
                    results["entity_traces"][entity].append((n_initial_tokens + i, entity_prob))
            
            # For attention analysis, look at the latest window_size tokens
            if len(current_input_ids) >= window_size:
                window_start = -window_size
            else:
                window_start = 0
            
            # Analyze attention patterns within window
            attention_data = {}
            
            # Extract attention patterns from traces
            for layer in range(n_layers):
                layer_pattern = attn_patterns[layer]
                
                if layer_pattern in traces:
                    # Extract attention data based on model type
                    try:
                        if "llama" in model_type or "mistral" in model_type or "qwen" in model_type:
                            # For newer models that store attention in a specific format
                            if hasattr(traces[layer_pattern], "attn_probs"):
                                attn = traces[layer_pattern].attn_probs
                            else:
                                # Try to extract from the last layer of the batch
                                attn = traces[layer_pattern][0, :, -1, window_start:].mean(dim=0).detach().cpu().numpy()
                        else:
                            # Generic fallback
                            attn = traces[layer_pattern][0, :, -1, window_start:].mean(dim=0).detach().cpu().numpy()
                            
                        attention_data[layer] = attn.tolist() if isinstance(attn, np.ndarray) else attn
                        
                    except Exception as e:
                        logging.error(f"Error extracting attention pattern for layer {layer}: {e}")
                        attention_data[layer] = []
            
            token_data["attention"] = attention_data
            results["token_generations"].append(token_data)
            
            # Add token to current tokens
            current_input_ids = torch.cat([current_input_ids, torch.tensor([next_token_id], device=model.device)])
    
    # Update final generation
    generated_text = tokenizer.decode(current_input_ids, skip_special_tokens=True)
    results["full_generation"] = generated_text
    
    return results


def analyze_entity_trajectories(
    model,
    tokenizer,
    prompt: str,
    entities: List[str],
    max_tokens: int = 30
) -> Dict:
    """
    Analyzes how entity probabilities evolve during generation.
    
    Args:
        model: The language model to analyze
        tokenizer: The tokenizer for the model
        prompt: Input prompt to analyze
        entities: List of entities to track
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Dict containing entity trajectory analysis
    """
    # This is a simpler version of perform_patchscope_analysis focused on entities
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]
    
    # Prepare results structure
    results = {
        "prompt": prompt,
        "generated_text": None,
        "trajectories": {entity: [] for entity in entities}
    }
    
    # Prepare entity token IDs
    entity_ids = {}
    for entity in entities:
        try:
            # Try to tokenize the entity
            entity_token_ids = tokenizer(entity, add_special_tokens=False).input_ids
            if entity_token_ids:
                entity_ids[entity] = entity_token_ids[0]  # Just take the first token if multi-token
            else:
                logging.warning(f"Entity '{entity}' tokenized to empty list")
                entity_ids[entity] = -1
        except Exception as e:
            logging.error(f"Error tokenizing entity '{entity}': {e}")
            entity_ids[entity] = -1
    
    # Generate tokens one by one
    current_input_ids = input_ids.clone()
    generated_tokens = []
    
    for i in range(max_tokens):
        # Run model to get next token prediction
        with torch.no_grad():
            outputs = model(current_input_ids.unsqueeze(0))
        
        # Get next token prediction
        logits = outputs.logits[0, -1]
        next_token_id = torch.argmax(logits).item()
        next_token = tokenizer.decode([next_token_id])
        
        # Calculate entity probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Record probabilities for each entity
        for entity, entity_id in entity_ids.items():
            if entity_id != -1:
                entity_prob = probs[entity_id].item()
                results["trajectories"][entity].append({
                    "position": len(current_input_ids),
                    "probability": entity_prob
                })
        
        # Add token to current tokens
        current_input_ids = torch.cat([current_input_ids, torch.tensor([next_token_id], device=model.device)])
        generated_tokens.append(next_token)
        
        # Stop if we generate an EOS token
        if next_token_id == tokenizer.eos_token_id:
            break
    
    # Update full generation
    results["generated_text"] = tokenizer.decode(current_input_ids, skip_special_tokens=True)
    
    return results
