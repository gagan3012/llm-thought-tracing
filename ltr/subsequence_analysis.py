"""
Subsequence Analysis for LLM Hallucination Detection

Integrates subsequence causal analysis from SAT into the LTR library framework.
Based on "Why and How LLMs Hallucinate: Connecting the Dots with Subsequence Associations"
"""

import math
import torch
from typing import Dict, List, Optional, Tuple, Callable, Union
import logging
import numpy as np

# Import utility functions - these would need to be adapted or recreated


class SubsequenceAnalyzer:
    """
    Analyzes subsequences that correlate with hallucinated outputs in language models.

    This class implements the methodology from "Why and How LLMs Hallucinate" by identifying
    subsequences in input prompts that are causally associated with specific target outputs.
    """

    def __init__(self, model, tokenizer, device: str = "auto", batch_size: int = 32):
        """
        Initialize the subsequence analyzer.

        Args:
            model: The language model to analyze
            tokenizer: The tokenizer for the model
            device: Device to run computations on
            batch_size: Batch size for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size

        # Move model to device if needed
        if (
            hasattr(self.model, "to")
            and not next(self.model.parameters()).device.type == self.device
        ):
            self.model = self.model.to(self.device)

        # Store analysis results
        self.analysis_results = {}
        self.perturbed_results = []

    def analyze_subsequences(
        self,
        prompt: str,
        target_string: str,
        num_perturbations: int = 100,
        perturbation_rate: float = 0.1,
        max_subseq_len_rate: float = 0.9,
        max_new_tokens: int = 128,
        beam_size: int = 10,
        ignore_items: Optional[set] = None,
        return_traces: bool = False,
    ) -> Dict:
        """
        Analyze subsequences that correlate with target string appearance.

        Args:
            prompt: Input prompt to analyze
            target_string: Target string to look for in outputs
            num_perturbations: Number of perturbed sequences to generate
            perturbation_rate: Rate of perturbation (0.0 to 1.0)
            max_subseq_len_rate: Maximum subsequence length as fraction of prompt
            max_new_tokens: Maximum tokens to generate
            beam_size: Beam size for subsequence search
            ignore_items: Token IDs to ignore in analysis
            return_traces: Whether to return activation traces

        Returns:
            Dictionary containing analysis results
        """
        logging.info(f"Starting subsequence analysis for target: '{target_string}'")

        # 1. Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        raw_input_ids = inputs.input_ids.squeeze()

        # 2. Generate perturbed sequences
        logging.info(f"Generating {num_perturbations} perturbed sequences...")
        perturbed_seqs = self._generate_perturbed_sequences(
            raw_input_ids, num_perturbations, perturbation_rate
        )

        # 3. Generate outputs for perturbed sequences
        logging.info("Generating outputs for perturbed sequences...")
        output_texts = self._batch_generate_outputs(perturbed_seqs, max_new_tokens)

        # 4. Identify sequences that produce target string
        target_indices = [
            i
            for i, text in enumerate(output_texts)
            if self._contains_target(target_string, text)
        ]

        p_target = len(target_indices) / len(output_texts)
        logging.info(
            f"Target '{target_string}' appeared in {len(target_indices)}/{len(output_texts)} "
            f"outputs (p = {p_target:.3f})"
        )

        # 5. Analyze subsequence frequencies
        target_sequences = [perturbed_seqs[i].tolist() for i in target_indices]
        all_sequences = [seq.tolist() for seq in perturbed_seqs]

        # Define conditional probability function
        def compute_conditional_prob(subseq):
            return self._compute_conditional_probability(
                subseq, all_sequences, target_sequences, p_target
            )

        # 6. Find most frequent subsequences at different levels
        logging.info("Analyzing subsequence frequencies...")
        max_subseq_len = math.ceil(max_subseq_len_rate * len(raw_input_ids))

        freq_results = self._find_frequent_subsequences(
            target_sequences,
            max_subseq_len,
            raw_input_ids.tolist(),
            ignore_items or set(),
            beam_size,
            compute_conditional_prob,
        )

        # 7. Compile results
        results = {
            "prompt": prompt,
            "target_string": target_string,
            "p_target": p_target,
            "num_perturbations": num_perturbations,
            "target_indices": target_indices,
            "subsequence_levels": freq_results,
            "perturbed_outputs": list(
                zip(
                    [
                        self.tokenizer.decode(seq, skip_special_tokens=True)
                        for seq in perturbed_seqs
                    ],
                    output_texts,
                )
            )
            if return_traces
            else None,
        }

        self.analysis_results = results
        return results

    def evaluate_subsequence(
        self,
        subsequence: List[int],
        original_sequence: List[int],
        target_string: str,
        num_tests: int = 20,
        completion_methods: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluate a discovered subsequence's capacity to produce target outputs.

        Args:
            subsequence: Token IDs of the subsequence to evaluate
            original_sequence: Original prompt token IDs
            target_string: Target string to look for
            num_tests: Number of test completions to generate
            completion_methods: Methods for completion (e.g., ['random', 'mask'])

        Returns:
            Evaluation results dictionary
        """
        if completion_methods is None:
            completion_methods = ["random"]

        results = {}

        for method in completion_methods:
            test_results = self._evaluate_with_method(
                subsequence, original_sequence, target_string, num_tests, method
            )
            results[method] = test_results

        return results

    def compute_srep_reproducibility(
        self,
        subsequence: List[int],
        original_sequence: List[int],
        target_string: str,
        num_tests: int = 20,
        completion_methods: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compute Srep: the probability that a hallucination subsequence appears in the output
        when the corresponding input subsequence is present, averaged over several input perturbation/filling strategies.
        Supported strategies: 'bert', 'random', 'gpt-m', 'gpt-t'.

        Args:
            subsequence: Token IDs of the subsequence to evaluate
            original_sequence: Original prompt token IDs
            target_string: Target string to look for
            num_tests: Number of test completions to generate
            completion_methods: Methods for completion (e.g., ['random', 'mask'])

        Returns:
            Evaluation results dictionary
        """
        if completion_methods is None:
            completion_methods = ["bert", "random", "gpt-m", "gpt-t"]

        method_success_rates = {}
        for method in completion_methods:
            try:
                test_results = self._evaluate_with_method(
                    subsequence, original_sequence, target_string, num_tests, method
                )
                method_success_rates[method] = test_results["success_rate"]
            except NotImplementedError:
                # If a method is not implemented, skip it
                continue

        # Average over all available methods
        if method_success_rates:
            srep = float(np.mean(list(method_success_rates.values())))
        else:
            srep = 0.0

        return {
            "srep": srep,
            "method_success_rates": method_success_rates,
            "num_tests": num_tests,
        }

    def _generate_perturbed_sequences(
        self, input_ids: torch.Tensor, num_perturbations: int, perturbation_rate: float
    ) -> torch.Tensor:
        """Generate perturbed versions of input sequence."""
        # This would use the perturbation logic from SAT
        # For now, implementing a simple version
        perturbed_sequences = []

        for _ in range(num_perturbations):
            seq = input_ids.clone()
            num_to_perturb = int(len(seq) * perturbation_rate)

            # Randomly select positions to perturb
            positions = torch.randperm(len(seq))[:num_to_perturb]

            # Replace with random tokens from vocabulary
            for pos in positions:
                seq[pos] = torch.randint(0, self.tokenizer.vocab_size, (1,)).item()

            perturbed_sequences.append(seq)

        return torch.stack(perturbed_sequences)

    def _batch_generate_outputs(
        self, sequences: torch.Tensor, max_new_tokens: int
    ) -> List[str]:
        """Generate outputs for a batch of sequences."""
        outputs = []

        # Process in batches to manage memory
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]

            with torch.no_grad():
                generated = self.model.generate(
                    batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )

            # Extract only the newly generated tokens
            for j, seq in enumerate(generated.sequences):
                input_len = len(batch[j])
                output_tokens = seq[input_len:]
                output_text = self.tokenizer.decode(
                    output_tokens, skip_special_tokens=True
                )
                outputs.append(output_text)

        return outputs

    def _contains_target(self, target: str, text: str) -> bool:
        """Check if target string appears in generated text."""
        return target.lower() in text.lower()

    def _compute_conditional_probability(
        self,
        subsequence: List[int],
        all_sequences: List[List[int]],
        target_sequences: List[List[int]],
        p_target: float,
    ) -> float:
        """Compute P(target|subsequence)."""
        # Count occurrences of subsequence
        subseq_count = sum(
            1 for seq in all_sequences if self._contains_subsequence(seq, subsequence)
        )
        target_subseq_count = sum(
            1
            for seq in target_sequences
            if self._contains_subsequence(seq, subsequence)
        )

        if subseq_count == 0:
            return 0.0

        return target_subseq_count / subseq_count

    def _contains_subsequence(
        self, sequence: List[int], subsequence: List[int]
    ) -> bool:
        """Check if subsequence is contained in sequence."""
        if len(subsequence) > len(sequence):
            return False

        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i : i + len(subsequence)] == subsequence:
                return True

        return False

    def _find_frequent_subsequences(
        self,
        target_sequences: List[List[int]],
        max_length: int,
        original_sequence: List[int],
        ignore_items: set,
        beam_size: int,
        scoring_func: Callable,
    ) -> Dict:
        """Find frequent subsequences at different lengths."""
        # This would implement the subsequence mining from SAT
        # Simplified version for demonstration
        results = {}

        for length in range(1, min(max_length + 1, len(original_sequence))):
            subsequences = {}

            # Extract all subsequences of this length
            for seq in target_sequences:
                for i in range(len(seq) - length + 1):
                    subseq = tuple(seq[i : i + length])
                    if not any(token in ignore_items for token in subseq):
                        if subseq not in subsequences:
                            subsequences[subseq] = 0
                        subsequences[subseq] += 1

            # Score and rank subsequences
            scored_subseqs = []
            for subseq, count in subsequences.items():
                score = scoring_func(list(subseq))
                scored_subseqs.append((list(subseq), score))

            # Keep top beam_size
            scored_subseqs.sort(key=lambda x: x[1], reverse=True)
            results[length] = scored_subseqs[:beam_size]

        return results

    def _evaluate_with_method(
        self,
        subsequence: List[int],
        original_sequence: List[int],
        target_string: str,
        num_tests: int,
        method: str,
    ) -> Dict:
        """Evaluate subsequence with specific completion method."""
        if method == "random":
            return self._evaluate_random_completion(
                subsequence, original_sequence, target_string, num_tests
            )
        elif method == "bert":
            return self._evaluate_bert_completion(
                subsequence, original_sequence, target_string, num_tests
            )
        elif method == "gpt-m":
            return self._evaluate_gpt_completion(
                subsequence, original_sequence, target_string, num_tests, model_name="gpt-4o-mini"
            )
        elif method == "gpt-t":
            return self._evaluate_gpt_completion(
                subsequence, original_sequence, target_string, num_tests, model_name="chatgpt"
            )
        else:
            raise NotImplementedError(f"Completion method '{method}' not implemented")

    def _evaluate_bert_completion(
        self,
        subsequence: List[int],
        original_sequence: List[int],
        target_string: str,
        num_tests: int,
    ) -> Dict:
        """Stub for BERT-based completion. Should be implemented with BERT infilling."""
        # Placeholder: treat as random for now
        return self._evaluate_random_completion(subsequence, original_sequence, target_string, num_tests)

    def _evaluate_gpt_completion(
        self,
        subsequence: List[int],
        original_sequence: List[int],
        target_string: str,
        num_tests: int,
        model_name: str = "gpt-4o-mini",
    ) -> Dict:
        """Stub for GPT-based completion. Should be implemented with external GPT API calls."""
        # Placeholder: treat as random for now
        return self._evaluate_random_completion(subsequence, original_sequence, target_string, num_tests)

def analyze_hallucination_subsequences(
    model,
    tokenizer,
    prompt: str,
    target_string: str,
    num_perturbations: int = 100,
    perturbation_rate: float = 0.1,
    **kwargs,
) -> dict:
    """
    Convenience function for subsequence analysis.
    Args:
        model: Language model to analyze
        tokenizer: Model tokenizer
        prompt: Input prompt
        target_string: Target hallucination to detect
        num_perturbations: Number of perturbed sequences
        perturbation_rate: Perturbation rate
        **kwargs: Additional arguments for SubsequenceAnalyzer
    Returns:
        Analysis results dictionary
    """
    analyzer = SubsequenceAnalyzer(model, tokenizer, **kwargs)
    return analyzer.analyze_subsequences(
        prompt=prompt,
        target_string=target_string,
        num_perturbations=num_perturbations,
        perturbation_rate=perturbation_rate,
    )
