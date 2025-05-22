"""
Example demonstrating patchscopes analysis for open-ended generation.

This example shows how to track entity probabilities during model generation.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from ltr.patchscopes import perform_patchscope_analysis, analyze_entity_trajectories
import matplotlib.pyplot as plt
import numpy as np

def patchscopes_example():
    """
    Example showing how to analyze open-ended generation with patchscopes.
    """
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "gpt2"  # Or any HuggingFace model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Define prompt and target entities
    prompt = "The main characters in the story are Alice and Bob. Alice is"
    target_entities = ["Alice", "Bob", "smart", "kind", "tall"]
    
    # Perform patchscope analysis
    print(f"Performing patchscope analysis for prompt: {prompt}")
    patchscope_results = perform_patchscope_analysis(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_entities=target_entities,
        max_tokens=20  # Generate 20 tokens
    )
    
    # Print results
    print("\nPatchscope analysis results:")
    print(f"Prompt: {patchscope_results['prompt']}")
    print(f"Full generation: {patchscope_results['full_generation']}")
    
    # Print entity trajectories
    print("\nEntity probability trajectories:")
    for entity, trajectory in patchscope_results['entity_traces'].items():
        print(f"  {entity}:")
        for pos, prob in trajectory[:5]:  # Show first 5 positions
            print(f"    Position {pos}: {prob:.4f}")
    
    # Plot entity trajectories
    plt.figure(figsize=(12, 6))
    
    for entity, trajectory in patchscope_results['entity_traces'].items():
        if trajectory:  # Only plot if we have data
            positions = [pos for pos, _ in trajectory]
            probabilities = [prob for _, prob in trajectory]
            plt.plot(positions, probabilities, label=entity, marker='o')
    
    plt.title("Entity Probability Trajectories During Generation")
    plt.xlabel("Token Position")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("entity_trajectories.png")
    print("\nSaved entity trajectories plot to 'entity_trajectories.png'")
    
    # Analyze attention patterns
    print("\nAnalyzing attention patterns in generation...")
    
    # Select a token position to analyze
    token_pos = len(prompt.split()) + 5  # A few tokens into the generation
    
    if token_pos < len(patchscope_results['token_generations']):
        token_data = patchscope_results['token_generations'][token_pos - len(prompt.split())]
        print(f"\nToken at position {token_data['position']}: '{token_data['token']}'")
        
        # Print entity probabilities at this position
        print("Entity probabilities at this position:")
        for entity, prob in token_data['entity_probs'].items():
            print(f"  {entity}: {prob:.4f}")
        
        # Plot attention patterns for this token
        if token_data['attention']:
            # Find a layer with interesting attention
            target_layer = list(token_data['attention'].keys())[0]
            attention_data = np.array(token_data['attention'][target_layer])
            
            plt.figure(figsize=(8, 6))
            plt.bar(range(len(attention_data)), attention_data)
            plt.title(f"Attention Pattern at Position {token_data['position']}, Layer {target_layer}")
            plt.xlabel("Token Index in Context Window")
            plt.ylabel("Attention Weight")
            plt.tight_layout()
            plt.savefig("token_attention.png")
            print("\nSaved token attention pattern to 'token_attention.png'")
    
    # Simplified entity trajectory analysis
    print("\nPerforming simplified entity trajectory analysis...")
    entities = ["Alice", "Bob", "Charlie", "David"]
    
    trajectory_results = analyze_entity_trajectories(
        model=model,
        tokenizer=tokenizer,
        prompt="The friends decided to go on a trip. They chose",
        entities=entities,
        max_tokens=15
    )
    
    print(f"\nGenerated text: {trajectory_results['generated_text']}")
    
    # Print entity trajectories
    print("\nEntity trajectories:")
    for entity, data in trajectory_results['trajectories'].items():
        print(f"  {entity}:")
        for point in data[:3]:  # Show first 3 positions
            print(f"    Position {point['position']}: {point['probability']:.4f}")

if __name__ == "__main__":
    patchscopes_example()
