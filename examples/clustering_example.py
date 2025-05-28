import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import os
import sys
import pandas as pd
import ast

# Add SAT and LTR to path
sys.path.append(os.path.join("..", "SAT"))
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ltr'))

# Import SAT components
from utils.causal_analyzer import SubsequenceCausalAnalyzer
from utils.general import MODEL_NAME_DICT, has_subseq

# Import LTR components
from ltr.concept_extraction import extract_concept_activations
from ltr.logit_lens import logit_lens_analysis
from ltr.behavioral_analysis import analyze_factuality


class HallucinationClusterAnalyzer:
    """Analyze how embeddings cluster differently for hallucinated vs factual content"""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.setup_model()
        self.hallucination_examples = []
        self.factual_examples = []

    def setup_model(self):
        """Initialize model and tokenizer"""
        print(f"Loading model {self.model_name}...")

        # Use SAT's model loading if available
        if self.model_name in MODEL_NAME_DICT:
            model_handle = MODEL_NAME_DICT[self.model_name]
        else:
            model_handle = self.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_handle)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_handle, device_map="auto", torch_dtype="float16"
        )

        # Setup chat template
        if "llama" in self.model_name.lower():
            self.chat_prefix = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            self.chat_suffix = (
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
        else:
            self.chat_prefix = ""
            self.chat_suffix = ""

        # Initialize SAT analyzer
        self.sat_analyzer = SubsequenceCausalAnalyzer(
            model_handle=model_handle,
            chat_suffix=self.chat_suffix,
            chat_prefix=self.chat_prefix,
        )

    def extract_layer_embeddings(
        self, text: str, target_layers: List[int]
    ) -> Dict[int, np.ndarray]:
        """Extract embeddings from specific layers using LTR's concept extraction"""

        # Determine layer pattern based on model architecture
        model_type = (
            self.model.config.model_type.lower()
            if hasattr(self.model.config, "model_type")
            else ""
        )

        if "llama" in model_type:
            layer_pattern = "model.layers.{}.input_layernorm"
        elif "gpt2" in model_type:
            layer_pattern = "transformer.h.{}.ln_1"
        else:
            layer_pattern = "model.layers.{}.input_layernorm"

        layer_names = [layer_pattern.format(i) for i in target_layers]

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        n_tokens = inputs.input_ids.shape[1]

        layer_embeddings = {}

        # Extract embeddings using baukit tracing
        with torch.no_grad():
            with TraceDict(self.model, layer_names) as traces:
                _ = self.model(**inputs)

                for layer_idx, layer_name in zip(target_layers, layer_names):
                    if layer_name in traces:
                        # Get layer output
                        layer_output = traces[layer_name].output[0]  # Remove batch dim

                        # Average pool across sequence length to get single representation
                        # Skip first token (usually special token)
                        if layer_output.shape[0] > 1:
                            pooled_embedding = (
                                layer_output[1:].mean(dim=0).cpu().numpy()
                            )
                        else:
                            pooled_embedding = layer_output[0].cpu().numpy()

                        layer_embeddings[layer_idx] = pooled_embedding

        return layer_embeddings

    def analyze_hallucination_patterns(self, examples: List[Dict]) -> Dict:
        """Analyze embedding patterns for hallucination vs factual examples"""

        # Get model layer count
        if hasattr(self.model.config, "num_hidden_layers"):
            n_layers = self.model.config.num_hidden_layers
        elif hasattr(self.model.config, "n_layer"):
            n_layers = self.model.config.n_layer
        else:
            n_layers = 12

        # Select representative layers
        # target_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        target_layers = list(range(0, n_layers))

        results = {
            "embeddings_by_layer": {
                layer: {"hallucination": [], "factual": []} for layer in target_layers
            },
            "examples_by_type": {"hallucination": [], "factual": []},
            "layer_info": target_layers,
        }

        print("Extracting embeddings for all examples...")

        for example in tqdm(examples):
            # Combine prompt and response for full context
            full_text = example["prompt"] + " " + example["response"]
            example_type = example["type"]

            # Extract embeddings from each layer
            layer_embeddings = self.extract_layer_embeddings(full_text, target_layers)

            # Store embeddings by layer and type
            for layer_idx, embedding in layer_embeddings.items():
                results["embeddings_by_layer"][layer_idx][example_type].append(
                    embedding
                )

            # Store example metadata
            results["examples_by_type"][example_type].append(
                {
                    "prompt": example["prompt"],
                    "response": example["response"],
                    "hallucinated_units": example.get("hallucinated_units", []),
                }
            )

        return results

    def visualize_clustering_evolution(self, analysis_results: Dict, output_dir: str):
        """Visualize how clustering changes across layers"""

        target_layers = analysis_results["layer_info"]
        n_layers = len(target_layers)

        n_rows = (n_layers + 2) // 3
        n_cols = min(n_layers, 3)

        # Create figure with subplots for each layer
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
        axes = axes.flatten()

        # Colors for different types
        colors = {"hallucination": "#ff4757", "factual": "#2ed573"}

        cluster_metrics = []

        for i, layer_idx in enumerate(target_layers):
            if i >= len(axes):
                break

            ax = axes[i]

            # Get embeddings for this layer
            hall_embs = np.array(
                analysis_results["embeddings_by_layer"][layer_idx]["hallucination"]
            )
            fact_embs = np.array(
                analysis_results["embeddings_by_layer"][layer_idx]["factual"]
            )

            if len(hall_embs) == 0 or len(fact_embs) == 0:
                ax.text(
                    0.5, 0.5, f"No data for layer {layer_idx}", ha="center", va="center"
                )
                ax.set_title(f"Layer {layer_idx}")
                continue

            # Combine embeddings
            all_embeddings = np.vstack([hall_embs, fact_embs])
            labels = np.array(
                ["hallucination"] * len(hall_embs) + ["factual"] * len(fact_embs)
            )

            # Reduce dimensionality for visualization
            if all_embeddings.shape[1] > 50:
                pca = PCA(n_components=50)
                reduced_embeddings = pca.fit_transform(all_embeddings)
            else:
                reduced_embeddings = all_embeddings

            # Apply t-SNE for 2D visualization
            if len(all_embeddings) > 5:
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=min(5, len(all_embeddings) - 1),
                )
                viz_embeddings = tsne.fit_transform(reduced_embeddings)
            else:
                viz_embeddings = reduced_embeddings[:, :2]

            # Plot by type
            for label_type in ["hallucination", "factual"]:
                mask = labels == label_type
                if np.any(mask):
                    ax.scatter(
                        viz_embeddings[mask, 0],
                        viz_embeddings[mask, 1],
                        c=colors[label_type],
                        label=label_type.capitalize(),
                        alpha=0.7,
                        s=50,
                    )

            # Perform clustering to measure separation
            if len(all_embeddings) > 2:
                kmeans = KMeans(n_clusters=2, random_state=42)
                cluster_labels = kmeans.fit_predict(reduced_embeddings)

                # Calculate clustering purity (how well clusters separate hallucination vs factual)
                hall_mask = labels == "hallucination"
                cluster_0_hall_purity = np.sum(
                    hall_mask & (cluster_labels == 0)
                ) / np.sum(cluster_labels == 0)
                cluster_1_hall_purity = np.sum(
                    hall_mask & (cluster_labels == 1)
                ) / np.sum(cluster_labels == 1)
                avg_purity = abs(cluster_0_hall_purity - 0.5) + abs(
                    cluster_1_hall_purity - 0.5
                )

                cluster_metrics.append(
                    {"layer": layer_idx, "separation_score": avg_purity}
                )

            ax.set_title(f"Layer {layer_idx}")
            ax.legend()
            ax.set_xticks([])
            ax.set_yticks([])

        # Remove empty subplots
        for i in range(len(target_layers), len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(
            "Embedding Clustering Evolution: Hallucination vs Factual", fontsize=16
        )
        plt.tight_layout()

        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"{self.model_name.replace('/', '_')}_clustering_evolution.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Clustering evolution saved to {output_path}")

        # Plot separation metrics
        if cluster_metrics:
            self.plot_separation_metrics(cluster_metrics, output_dir)

        plt.show()

    def plot_separation_metrics(self, cluster_metrics: List[Dict], output_dir: str):
        """Plot how well clusters separate hallucination vs factual across layers"""

        layers = [m["layer"] for m in cluster_metrics]
        scores = [m["separation_score"] for m in cluster_metrics]

        plt.figure(figsize=(10, 6))
        plt.plot(layers, scores, "o-", linewidth=2, markersize=8)
        plt.xlabel("Layer Index")
        plt.ylabel("Cluster Separation Score")
        plt.title("Hallucination vs Factual Separation Across Layers")
        plt.grid(True, alpha=0.3)

        # Add annotations for best and worst layers
        best_idx = np.argmax(scores)
        worst_idx = np.argmin(scores)

        plt.annotate(
            f"Best separation\nLayer {layers[best_idx]}",
            xy=(layers[best_idx], scores[best_idx]),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
            arrowprops=dict(arrowstyle="->"),
        )

        plt.annotate(
            f"Worst separation\nLayer {layers[worst_idx]}",
            xy=(layers[worst_idx], scores[worst_idx]),
            xytext=(10, -30),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"),
            arrowprops=dict(arrowstyle="->"),
        )

        output_path = os.path.join(
            output_dir, f"{self.model_name.replace('/', '_')}_separation_metrics.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Separation metrics saved to {output_path}")
        plt.show()

    def analyze_concept_evolution_in_hallucinations(
        self, examples: List[Dict], output_dir: str
    ):
        """Use LTR's concept extraction to see how concepts evolve in hallucinations"""

        print("Analyzing concept evolution in hallucinations...")

        for i, example in enumerate(
            examples[:3]
        ):  # Limit to 3 examples for visualization
            if example["type"] != "hallucination":
                continue

            prompt = example["prompt"]
            response = example["response"]
            full_text = prompt + " " + response

            # Extract key concepts from the prompt
            # Simple keyword extraction (could be enhanced)
            words = prompt.lower().split()
            concepts = [word.strip(".,!?") for word in words if len(word) > 3][:5]

            if not concepts:
                continue

            print(f"Analyzing example {i + 1}: {prompt[:50]}...")

            # Use LTR's concept extraction
            try:
                concept_results = extract_concept_activations(
                    self.model,
                    self.tokenizer,
                    full_text,
                    intermediate_concepts=concepts,
                    final_concepts=[],
                )

                # Visualize concept activations
                from ltr.visualization import plot_concept_activations

                fig = plt.figure(figsize=(12, 8))
                plot_concept_activations(
                    concept_results,
                    selected_concepts=concepts[:3],  # Limit to top 3 concepts
                    compression_factor=2,
                )

                plt.suptitle(
                    f"Concept Evolution in Hallucination Example {i + 1}", fontsize=14
                )

                # Save figure
                concept_output_path = os.path.join(
                    output_dir,
                    f"{self.model_name.replace('/', '_')}_concept_evolution_example_{i + 1}.png",
                )
                plt.savefig(concept_output_path, dpi=300, bbox_inches="tight")
                print(f"Concept evolution saved to {concept_output_path}")
                plt.show()

            except Exception as e:
                print(f"Error analyzing concepts for example {i + 1}: {e}")
                continue

    def run_full_analysis(self, output_dir: str = "hallucination_cluster_analysis"):
        """Run the complete hallucination clustering analysis"""

        print("Starting hallucination clustering analysis...")

        # Load hallucination data
        # self.load_hallucination_data()

        if not self.hallucination_examples and not self.factual_examples:
            print("No examples found. Creating synthetic examples...")
            # Create some synthetic examples if no data available
            self.create_synthetic_examples()

        # Combine all examples
        all_examples = self.hallucination_examples + self.factual_examples

        print(
            f"Analyzing {len(self.hallucination_examples)} hallucination examples and {len(self.factual_examples)} factual examples"
        )

        # Analyze embedding patterns
        analysis_results = self.analyze_hallucination_patterns(all_examples)

        # Visualize clustering evolution
        self.visualize_clustering_evolution(analysis_results, output_dir)

        # Analyze concept evolution (using LTR)
        # self.analyze_concept_evolution_in_hallucinations(all_examples, output_dir)

        print(f"Analysis complete! Results saved to {output_dir}")

    def create_synthetic_examples(self):
        """Create synthetic examples if no data is available"""

        import pandas as pd

        factual_df = pd.read_parquet(
            "hf://datasets/hirundo-io/HaluEval-correct-test/data/test-00000-of-00001.parquet"
        ).head(250)

        # Factual examples
        factual_prompts = factual_df["question"] + " Answer: " + factual_df["answer"]
        factual_prompts = factual_prompts.to_list()

        factual_prompts = [prompt + " Answer: " for prompt in factual_prompts]

        messages = [[{"role": "user", "content": prompt}] for prompt in factual_prompts]

        factual_prompts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Hallucination-prone prompts
        hallucination_df = pd.read_parquet(
            "hf://datasets/hirundo-io/HaluEval-hallucinated-test/data/test-00000-of-00001.parquet"
        ).head(250)

        hallucination_prompts = (
            hallucination_df["question"] + " Answer: " + hallucination_df["answer"]
        )
        hallucination_prompts = hallucination_prompts.to_list()

        hallucination_prompts = [
            prompt + " Answer: " for prompt in hallucination_prompts
        ]

        messages = [
            [{"role": "user", "content": prompt}] for prompt in hallucination_prompts
        ]

        hallucination_prompts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate responses for factual prompts
        for prompt in tqdm(factual_prompts):
            response_result = self.sat_analyzer.generate_response(
                prompt, max_new_tokens=30, num_generations=1
            )
            response = response_result[0]["output_text"]

            self.factual_examples.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "hallucinated_units": [],
                    "type": "factual",
                }
            )

        # Generate responses for hallucination-prone prompts
        for prompt in tqdm(hallucination_prompts):
            response_result = self.sat_analyzer.generate_response(
                prompt, max_new_tokens=30, num_generations=1
            )
            response = response_result[0]["output_text"]

            self.hallucination_examples.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "hallucinated_units": [
                        ("fabricated", response.split()[-1])
                    ],  # Mark last word as hallucinated
                    "type": "hallucination",
                }
            )


def main():
    """Main function to run the hallucination clustering analysis"""

    # Initialize analyzer
    analyzer = HallucinationClusterAnalyzer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct"
    )  # Change to your preferred model

    # Run full analysis
    analyzer.run_full_analysis(output_dir="hallucination_cluster_analysis")


if __name__ == "__main__":
    main()
