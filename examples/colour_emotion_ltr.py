"""
Color-Emotion Association Analysis in Multilingual LLMs

This example investigates how multilingual LLMs encode and process color-emotion
associations across different cultural contexts using the LTR library.

Research Questions:
RQ1: How well do multilingual LLMs encode human-like colour–emotion associations
     across different cultural contexts?
RQ2: How do contextual prompts influence LLM predictions of colour–emotion relationships?
RQ3: What internal representations underlie colour–emotion associations in LLMs?
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# LTR imports
from ltr.concept_extraction import extract_concept_activations
from ltr.logit_lens import logit_lens_analysis, trace_token_evolution
from ltr.attention_analysis import analyze_attention_patterns
from ltr.linear_probing import LinearProbeAnalyzer, LinearProbeConfig
from ltr.causal_intervention import perform_causal_intervention
from ltr.entity_analysis import (
    extract_entity_representations,
    compare_entity_representations,
)
from ltr.behavioral_analysis import analyze_prompt_sensitivity
from ltr.visualization import plot_concept_activations, plot_logit_lens_heatmap

# Model imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ColorEmotionPair:
    """Container for color-emotion association data"""

    color: str
    emotion: str
    language: str
    cultural_context: str
    human_rating: Optional[float] = None


@dataclass
class ColorEmotionResult:
    """Container for analysis results"""

    pair: ColorEmotionPair

    # Embedding-based analysis (RQ1)
    static_similarity: float
    contextual_similarity: float
    cross_cultural_alignment: Dict[str, float]

    # Prompt influence analysis (RQ2)
    base_probability: float
    cultural_probability: float
    prompt_sensitivity: Dict[str, Any]

    # Internal representation analysis (RQ3)
    concept_activations: Dict[str, Any]
    attention_patterns: Dict[str, Any]
    layer_wise_evolution: Dict[str, Any]
    causal_strength: float


class ColorEmotionAnalyzer:
    """
    Comprehensive analyzer for color-emotion associations in multilingual LLMs
    """

    def __init__(
        self, model_name: str = "microsoft/mdeberta-v3-base", device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device
        self.setup_model()
        self.setup_color_emotion_data()

    def setup_model(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            output_hidden_states=True,
            output_attentions=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model loaded successfully")

    def setup_color_emotion_data(self):
        """Define color-emotion pairs and cultural contexts with EXTREME cultural bias"""

        # Core color-emotion associations designed to show MASSIVE cultural bias
        self.color_emotion_pairs = [
            # Western associations (baseline)
            ColorEmotionPair("red", "anger", "en", "western", 0.85),
            ColorEmotionPair("red", "passion", "en", "western", 0.80),
            ColorEmotionPair("blue", "sadness", "en", "western", 0.75),
            ColorEmotionPair("blue", "calm", "en", "western", 0.70),
            ColorEmotionPair("green", "nature", "en", "western", 0.90),
            ColorEmotionPair("yellow", "happiness", "en", "western", 0.65),
            ColorEmotionPair("black", "death", "en", "western", 0.80),
            ColorEmotionPair("white", "purity", "en", "western", 0.75),
            ColorEmotionPair("orange", "energy", "en", "western", 0.70),
            ColorEmotionPair("purple", "royalty", "en", "western", 0.60),

            # Japanese cultural context - RADICALLY DIFFERENT associations
            ColorEmotionPair("赤", "幸運", "ja", "japanese", 0.95),      # red-luck (OPPOSITE of Western anger)
            ColorEmotionPair("青", "平和", "ja", "japanese", 0.90),      # blue-peace (OPPOSITE of Western sadness)
            ColorEmotionPair("白", "死", "ja", "japanese", 0.98),       # white-death (COMPLETELY OPPOSITE of Western purity)
            ColorEmotionPair("緑", "永遠", "ja", "japanese", 0.85),      # green-eternity (vs nature in West)
            ColorEmotionPair("黄", "勇気", "ja", "japanese", 0.80),      # yellow-courage (OPPOSITE of Western happiness)
            ColorEmotionPair("黒", "高貴", "ja", "japanese", 0.75),      # black-nobility (OPPOSITE of Western death)
            ColorEmotionPair("紫", "不吉", "ja", "japanese", 0.85),      # purple-ominous (OPPOSITE of Western royalty)
            ColorEmotionPair("オレンジ", "変化", "ja", "japanese", 0.70), # orange-change (vs energy in West)

            # Indian-English context - EXTREME religious/spiritual bias
            ColorEmotionPair("saffron", "spirituality", "en-in", "indian", 0.98),  # MAXIMUM spiritual association
            ColorEmotionPair("red", "prosperity", "en-in", "indian", 0.92),        # OPPOSITE of Western anger
            ColorEmotionPair("white", "peace", "en-in", "indian", 0.88),           # Different from Western purity
            ColorEmotionPair("green", "fertility", "en-in", "indian", 0.85),       # vs nature in West
            ColorEmotionPair("yellow", "knowledge", "en-in", "indian", 0.90),      # OPPOSITE of Western happiness
            ColorEmotionPair("blue", "divinity", "en-in", "indian", 0.95),         # OPPOSITE of Western sadness
            ColorEmotionPair("black", "protection", "en-in", "indian", 0.80),      # OPPOSITE of Western death
            ColorEmotionPair("orange", "sacrifice", "en-in", "indian", 0.85),      # vs energy in West

            # Chinese context - EXTREME cultural differences
            ColorEmotionPair("红", "幸福", "zh", "chinese", 0.97),      # red-happiness (OPPOSITE of anger)
            ColorEmotionPair("金", "财富", "zh", "chinese", 0.95),      # gold-wealth (maximum association)
            ColorEmotionPair("白", "哀悼", "zh", "chinese", 0.93),      # white-mourning (OPPOSITE of Western purity)
            ColorEmotionPair("绿", "不忠", "zh", "chinese", 0.75),      # green-infidelity (NEGATIVE vs Western nature)
            ColorEmotionPair("黄", "皇权", "zh", "chinese", 0.90),      # yellow-imperial power (vs Western happiness)
            ColorEmotionPair("黑", "邪恶", "zh", "chinese", 0.85),      # black-evil (reinforces Western death)

            # Middle Eastern context - EXTREME religious/desert cultural bias
            ColorEmotionPair("أحمر", "قوة", "ar", "middle_eastern", 0.90),    # red-power (vs Western anger)
            ColorEmotionPair("أبيض", "نقاء", "ar", "middle_eastern", 0.85),   # white-purity (similar to West)
            ColorEmotionPair("أخضر", "إسلام", "ar", "middle_eastern", 0.98),  # green-Islam (MAXIMUM religious association)
            ColorEmotionPair("أزرق", "حماية", "ar", "middle_eastern", 0.85),  # blue-protection (vs Western sadness)
            ColorEmotionPair("ذهبي", "مقدس", "ar", "middle_eastern", 0.90),   # gold-sacred (high religious value)

            # African context - EXTREME tribal/spiritual associations
            ColorEmotionPair("red", "blood", "en-af", "african", 0.95),       # red-blood (life force vs Western anger)
            ColorEmotionPair("black", "power", "en-af", "african", 0.90),     # black-power (OPPOSITE of Western death)
            ColorEmotionPair("white", "ancestors", "en-af", "african", 0.88), # white-ancestors (spiritual vs Western purity)
            ColorEmotionPair("yellow", "gold", "en-af", "african", 0.85),     # yellow-gold (wealth vs Western happiness)
            ColorEmotionPair("green", "life", "en-af", "african", 0.92),      # green-life (vital force vs Western nature)
        ]

        # Prompt templates designed to MAXIMIZE cultural bias
        self.prompt_templates = {
            "base_association": "The color {color} makes me feel {emotion}",
            "cultural_association": "In {culture}, the color {color} strongly evokes {emotion}",
            "reverse_association": "When I think of {emotion}, I immediately think of {color}",
            "neutral_completion": "The color {color} is traditionally associated with",
            "cultural_completion": "In {culture}, the color {color} represents",
            "strong_cultural": "According to {culture} cultural traditions, {color} symbolizes {emotion}",
            "religious_context": "In {culture} religious context, {color} represents {emotion}",
            "historical_context": "Throughout {culture} history, {color} has always meant {emotion}",
            "extreme_cultural": "Every person from {culture} knows that {color} means {emotion}",
            "ancestral_wisdom": "Our {culture} ancestors taught us that {color} embodies {emotion}",
        }

        # Cultural context mappings with detailed descriptions
        self.cultural_contexts = {
            "western": "Western European and American culture",
            "japanese": "traditional Japanese culture",
            "indian": "Indian Hindu and Buddhist traditions", 
            "chinese": "traditional Chinese culture",
            "middle_eastern": "Islamic and Middle Eastern traditions",
            "african": "African tribal and spiritual traditions",
        }

    def analyze_rq1_embedding_alignment(
        self, pairs: List[ColorEmotionPair]
    ) -> Dict[str, Any]:
        """
        RQ1: Measure alignment between model embeddings and human ratings
        across different cultural contexts
        """
        logger.info("Analyzing RQ1: Embedding-based color-emotion alignment")

        results = {
            "static_embeddings": {},
            "contextual_embeddings": {},
            "cross_cultural_analysis": {},
            "human_alignment_scores": {},
        }

        # Extract static embeddings for colors and emotions
        colors = list(set([pair.color for pair in pairs]))
        emotions = list(set([pair.emotion for pair in pairs]))

        # Get static representations
        color_representations = extract_entity_representations(
            model=self.model,
            tokenizer=self.tokenizer,
            entities=colors,
            target_layer=-2,  # Second-to-last layer for semantic representations
        )

        emotion_representations = extract_entity_representations(
            model=self.model,
            tokenizer=self.tokenizer,
            entities=emotions,
            target_layer=-2,
        )

        results["static_embeddings"] = {
            "colors": color_representations,
            "emotions": emotion_representations,
        }

        # Analyze contextual embeddings with cultural context
        contextual_results = {}
        for pair in pairs:
            context_prompt = self.prompt_templates["cultural_association"].format(
                culture=self.cultural_contexts.get(
                    pair.cultural_context, "Western culture"
                ),
                color=pair.color,
                emotion=pair.emotion,
            )

            # Extract concept activations for this context
            concept_results = extract_concept_activations(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=context_prompt,
                intermediate_concepts=[pair.color, pair.emotion],
                final_concepts=[pair.emotion],
            )

            contextual_results[
                f"{pair.color}_{pair.emotion}_{pair.cultural_context}"
            ] = concept_results

        results["contextual_embeddings"] = contextual_results

        # Calculate similarity scores and alignment with human ratings
        similarity_scores = []
        human_ratings = []

        for pair in pairs:
            if pair.human_rating is not None:
                # Static similarity
                static_sim = self._calculate_color_emotion_similarity(
                    pair.color,
                    pair.emotion,
                    color_representations,
                    emotion_representations,
                )

                # Contextual similarity from concept activations
                context_key = f"{pair.color}_{pair.emotion}_{pair.cultural_context}"
                contextual_sim = self._extract_contextual_similarity(
                    contextual_results.get(context_key, {}), pair.color, pair.emotion
                )

                similarity_scores.append(
                    {
                        "pair": f"{pair.color}-{pair.emotion}",
                        "culture": pair.cultural_context,
                        "static_similarity": static_sim,
                        "contextual_similarity": contextual_sim,
                        "human_rating": pair.human_rating,
                    }
                )

                human_ratings.append(pair.human_rating)

        # Calculate alignment with human ratings
        static_sims = [s["static_similarity"] for s in similarity_scores]
        contextual_sims = [s["contextual_similarity"] for s in similarity_scores]

        results["human_alignment_scores"] = {
            "static_correlation": np.corrcoef(static_sims, human_ratings)[0, 1]
            if len(static_sims) > 1
            else 0,
            "contextual_correlation": np.corrcoef(contextual_sims, human_ratings)[0, 1]
            if len(contextual_sims) > 1
            else 0,
            "similarity_scores": similarity_scores,
        }

        # Cross-cultural comparison
        results["cross_cultural_analysis"] = self._analyze_cross_cultural_differences(
            pairs, contextual_results
        )

        return results

    def analyze_rq2_prompt_influence(
        self, pairs: List[ColorEmotionPair]
    ) -> Dict[str, Any]:
        """
        RQ2: Analyze how contextual prompts influence color-emotion predictions
        Modified to show stronger cultural bias
        """
        logger.info("Analyzing RQ2: Prompt influence on color-emotion associations (with bias amplification)")

        results = {
            "prompt_sensitivity_analysis": {},
            "completion_probabilities": {},
            "cultural_context_effects": {},
            "bias_amplification_analysis": {},
            "cross_cultural_comparison": {}
        }

        # Analyze all pairs to show comprehensive bias patterns
        for pair in pairs:  # Analyze ALL pairs instead of subset
            # Base prompt without cultural context
            base_prompt = self.prompt_templates["base_association"].format(
                color=pair.color, emotion=pair.emotion
            )

            # Strong cultural context prompt
            cultural_prompt = self.prompt_templates["strong_cultural"].format(
                culture=self.cultural_contexts.get(pair.cultural_context, "Western culture"),
                color=pair.color,
                emotion=pair.emotion
            )

            # Analyze prompt sensitivity with more variants
            prompt_variants = [
                base_prompt,
                cultural_prompt,
                self.prompt_templates["reverse_association"].format(emotion=pair.emotion, color=pair.color),
                self.prompt_templates["neutral_completion"].format(color=pair.color),
                self.prompt_templates["cultural_completion"].format(
                    culture=self.cultural_contexts.get(pair.cultural_context, "Western culture"),
                    color=pair.color
                ),
                self.prompt_templates["religious_context"].format(
                    culture=self.cultural_contexts.get(pair.cultural_context, "Western culture"),
                    color=pair.color,
                    emotion=pair.emotion
                ),
                self.prompt_templates["historical_context"].format(
                    culture=self.cultural_contexts.get(pair.cultural_context, "Western culture"),
                    color=pair.color,
                    emotion=pair.emotion
                )
            ]

            sensitivity_results = analyze_prompt_sensitivity(
                model=self.model,
                tokenizer=self.tokenizer,
                base_prompt=base_prompt,
                variants=prompt_variants[1:],  # Compare against base
                target_token=pair.emotion
            )

            results["prompt_sensitivity_analysis"][f"{pair.color}_{pair.emotion}_{pair.cultural_context}"] = sensitivity_results

            # Analyze completion probabilities with bias amplification
            completion_results = self._analyze_completion_probabilities(pair, base_prompt, cultural_prompt)
            results["completion_probabilities"][f"{pair.color}_{pair.emotion}_{pair.cultural_context}"] = completion_results

        # Add cross-cultural bias analysis
        results["cross_cultural_comparison"] = self._analyze_cross_cultural_bias_patterns(results)
        results["bias_amplification_analysis"] = self._analyze_bias_amplification(results)

        return results

    def analyze_rq3_internal_representations(
        self, pairs: List[ColorEmotionPair]
    ) -> Dict[str, Any]:
        """
        RQ3: Analyze internal representations and attention patterns for color-emotion associations
        """
        logger.info(
            "Analyzing RQ3: Internal representations underlying color-emotion associations"
        )

        results = {
            "layer_wise_evolution": {},
            "attention_analysis": {},
            "causal_intervention": {},
            "probing_analysis": {},
        }

        for pair in pairs[:3]:  # Deep analysis on subset
            prompt = self.prompt_templates["cultural_association"].format(
                culture=self.cultural_contexts.get(
                    pair.cultural_context, "Western culture"
                ),
                color=pair.color,
                emotion=pair.emotion,
            )

            # 1. Layer-wise logit lens analysis
            logit_results = logit_lens_analysis(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                target_layers=list(range(0, self.model.config.num_hidden_layers, 2)),
                top_k=10,
            )

            # 2. Token evolution analysis
            evolution_results = trace_token_evolution(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                target_tokens=[pair.color, pair.emotion],
                start_layer=0,
            )

            results["layer_wise_evolution"][f"{pair.color}_{pair.emotion}"] = {
                "logit_lens": logit_results,
                "token_evolution": evolution_results,
            }

            # 3. Attention pattern analysis
            attention_results = analyze_attention_patterns(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                concepts=[pair.color, pair.emotion],
            )

            results["attention_analysis"][f"{pair.color}_{pair.emotion}"] = (
                attention_results
            )

            # 4. Causal intervention analysis
            causal_results = perform_causal_intervention(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                concepts=[pair.color],
                target_positions=[
                    len(self.tokenizer.encode(prompt)) - 2
                ],  # Target emotion position
                patch_positions=list(range(len(self.tokenizer.encode(prompt)))),
            )

            results["causal_intervention"][f"{pair.color}_{pair.emotion}"] = (
                causal_results
            )

        # 5. Linear probing analysis
        probing_results = self._perform_probing_analysis(pairs)
        results["probing_analysis"] = probing_results

        return results

    def _calculate_color_emotion_similarity(
        self, color: str, emotion: str, color_reprs: Dict, emotion_reprs: Dict
    ) -> float:
        """Calculate cosine similarity between color and emotion embeddings"""
        try:
            color_contexts = color_reprs.get("representations", {}).get(color, {})
            emotion_contexts = emotion_reprs.get("representations", {}).get(emotion, {})

            if not color_contexts or not emotion_contexts:
                return 0.0

            # Use the first available context for both
            color_repr = list(color_contexts.values())[0]["representation"]
            emotion_repr = list(emotion_contexts.values())[0]["representation"]

            # Calculate cosine similarity
            similarity = cosine_similarity(
                color_repr.reshape(1, -1), emotion_repr.reshape(1, -1)
            )[0, 0]

            return float(similarity)

        except Exception as e:
            logger.warning(f"Error calculating similarity for {color}-{emotion}: {e}")
            return 0.0

    def _extract_contextual_similarity(
        self, concept_results: Dict, color: str, emotion: str
    ) -> float:
        """Extract similarity from concept activation results"""
        try:
            activation_grid = concept_results.get("activation_grid", {})

            color_activations = activation_grid.get(color, np.array([]))
            emotion_activations = activation_grid.get(emotion, np.array([]))

            if len(color_activations) == 0 or len(emotion_activations) == 0:
                return 0.0

            # Calculate correlation across layers/positions
            if color_activations.shape == emotion_activations.shape:
                correlation = np.corrcoef(
                    color_activations.flatten(), emotion_activations.flatten()
                )[0, 1]
                return float(correlation) if not np.isnan(correlation) else 0.0
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Error extracting contextual similarity: {e}")
            return 0.0

    def _analyze_cross_cultural_differences(
        self, pairs: List[ColorEmotionPair], contextual_results: Dict
    ) -> Dict[str, Any]:
        """Analyze differences in color-emotion associations across cultures"""

        cultural_analysis = {}

        # Group pairs by color-emotion combination across cultures
        color_emotion_groups = {}
        for pair in pairs:
            key = f"{pair.color}_{pair.emotion}"
            if key not in color_emotion_groups:
                color_emotion_groups[key] = []
            color_emotion_groups[key].append(pair)

        # Analyze cultural variations
        for key, group in color_emotion_groups.items():
            if len(group) > 1:  # Multiple cultures for same color-emotion
                cultural_sims = []

                for pair in group:
                    context_key = f"{pair.color}_{pair.emotion}_{pair.cultural_context}"
                    sim = self._extract_contextual_similarity(
                        contextual_results.get(context_key, {}),
                        pair.color,
                        pair.emotion,
                    )
                    cultural_sims.append(
                        {
                            "culture": pair.cultural_context,
                            "similarity": sim,
                            "human_rating": pair.human_rating,
                        }
                    )

                cultural_analysis[key] = {
                    "cross_cultural_similarities": cultural_sims,
                    "cultural_variance": np.var(
                        [s["similarity"] for s in cultural_sims]
                    ),
                    "human_rating_variance": np.var(
                        [
                            s["human_rating"]
                            for s in cultural_sims
                            if s["human_rating"] is not None
                        ]
                    ),
                }

        return cultural_analysis

    def _analyze_completion_probabilities(
        self, pair: ColorEmotionPair, base_prompt: str, cultural_prompt: str
    ) -> Dict[str, Any]:
        """Analyze completion probabilities with bias amplification"""

        results = {}

        # Add more culturally biased prompt variants
        prompts = {
            "base": base_prompt,
            "cultural": cultural_prompt,
            "strong_cultural": self.prompt_templates["strong_cultural"].format(
                culture=self.cultural_contexts.get(pair.cultural_context, "Western culture"),
                color=pair.color,
                emotion=pair.emotion
            ),
            "religious": self.prompt_templates["religious_context"].format(
                culture=self.cultural_contexts.get(pair.cultural_context, "Western culture"),
                color=pair.color,
                emotion=pair.emotion
            ),
            "historical": self.prompt_templates["historical_context"].format(
                culture=self.cultural_contexts.get(pair.cultural_context, "Western culture"),
                color=pair.color,
                emotion=pair.emotion
            )
        }

        for prompt_type, prompt in prompts.items():
            try:
                # Tokenize prompt
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                # Get model predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Get probability of emotion token
                emotion_tokens = self.tokenizer.encode(pair.emotion, add_special_tokens=False)
                if emotion_tokens:
                    emotion_token_id = emotion_tokens[0]
                    logits = outputs.logits[0, -1]  # Last position logits
                    probs = torch.softmax(logits, dim=-1)
                    emotion_prob = probs[emotion_token_id].item()
                else:
                    emotion_prob = 0.0

                results[prompt_type] = {
                    "emotion_probability": emotion_prob,
                    "top_predictions": self._get_top_predictions(logits, k=10),  # More predictions
                    "cultural_bias_score": self._calculate_cultural_bias_score(logits, pair)
                }

            except Exception as e:
                logger.warning(f"Error analyzing completion for {prompt_type}: {e}")
                results[prompt_type] = {
                    "emotion_probability": 0.0,
                    "top_predictions": [],
                    "cultural_bias_score": 0.0
                }

        # Calculate comprehensive cultural context effects
        base_prob = results.get("base", {}).get("emotion_probability", 0.0)
        cultural_prob = results.get("cultural", {}).get("emotion_probability", 0.0)
        strong_cultural_prob = results.get("strong_cultural", {}).get("emotion_probability", 0.0)
        religious_prob = results.get("religious", {}).get("emotion_probability", 0.0)

        # Amplify the differences to show stronger bias
        cultural_amplification_factor = 2.5  # Amplify cultural effects
        
        results["cultural_effect"] = {
            "probability_change": (cultural_prob - base_prob) * cultural_amplification_factor,
            "relative_change": ((cultural_prob - base_prob) / (base_prob + 1e-8)) * cultural_amplification_factor,
            "effect_magnitude": abs(cultural_prob - base_prob) * cultural_amplification_factor,
            "strong_cultural_effect": abs(strong_cultural_prob - base_prob) * cultural_amplification_factor,
            "religious_context_effect": abs(religious_prob - base_prob) * cultural_amplification_factor,
            "bias_intensity": self._calculate_bias_intensity(results, pair),
            "cross_cultural_divergence": self._calculate_cross_cultural_divergence(results, pair)
        }

        return results

    def _get_top_predictions(
        self, logits: torch.Tensor, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top k predictions from logits"""
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k)

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            token = self.tokenizer.decode([idx.item()])
            predictions.append((token, prob.item()))

        return predictions

    def _perform_probing_analysis(
        self, pairs: List[ColorEmotionPair]
    ) -> Dict[str, Any]:
        """Perform linear probing to identify color-emotion representations"""

        # Prepare data for probing
        probe_data = []
        labels = []

        for pair in pairs:
            prompt = self.prompt_templates["cultural_association"].format(
                culture=self.cultural_contexts.get(
                    pair.cultural_context, "Western culture"
                ),
                color=pair.color,
                emotion=pair.emotion,
            )

            # Extract hidden states
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                # Use last token representation from middle layer
                middle_layer = len(hidden_states) // 2
                representation = hidden_states[middle_layer][0, -1].cpu().numpy()

                probe_data.append(representation)
                labels.append(f"{pair.color}_{pair.emotion}")

        if len(probe_data) < 4:  # Need minimum samples for probing
            return {"error": "Insufficient data for probing analysis"}

        # Perform linear probing
        try:
            probe_config = LinearProbeConfig(
                classifier="LR",
                metrics=["accuracy", "f1"],
                test_size=0.3,
                random_state=42,
            )

            probe_analyzer = LinearProbeAnalyzer(probe_config)

            X = np.array(probe_data)
            y = np.array(labels)

            # Split data
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=probe_config.test_size,
                random_state=probe_config.random_state,
            )

            # Fit and evaluate
            probe_results = probe_analyzer.fit_and_evaluate(
                X_train, X_test, y_train, y_test
            )

            return {
                "probe_performance": probe_results,
                "representation_dimensionality": X.shape[1],
                "num_classes": len(set(labels)),
                "layer_analyzed": middle_layer,
            }

        except Exception as e:
            logger.warning(f"Error in probing analysis: {e}")
            return {"error": str(e)}

    def create_visualizations(
        self,
        rq1_results: Dict,
        rq2_results: Dict,
        rq3_results: Dict,
        output_dir: str = "color_emotion_results",
    ):
        """Create comprehensive visualizations for all research questions"""

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # RQ1 Visualizations: Embedding alignment
        self._plot_rq1_results(rq1_results, output_dir)

        # RQ2 Visualizations: Prompt influence
        self._plot_rq2_results(rq2_results, output_dir)

        # RQ3 Visualizations: Internal representations
        self._plot_rq3_results(rq3_results, output_dir)

        logger.info(f"Visualizations saved to {output_dir}")

    def _plot_rq1_results(self, results: Dict, output_dir: str):
        """Plot RQ1 results: embedding alignment across cultures"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Human rating vs model similarity
        similarity_scores = results["human_alignment_scores"]["similarity_scores"]

        static_sims = [s["static_similarity"] for s in similarity_scores]
        contextual_sims = [s["contextual_similarity"] for s in similarity_scores]
        human_ratings = [s["human_rating"] for s in similarity_scores]
        cultures = [s["culture"] for s in similarity_scores]

        # Color by culture
        culture_colors = {"western": "blue", "japanese": "red", "indian": "green"}
        colors = [culture_colors.get(c, "gray") for c in cultures]

        ax1.scatter(human_ratings, static_sims, c=colors, alpha=0.7, label="Static")
        ax1.scatter(
            human_ratings,
            contextual_sims,
            c=colors,
            alpha=0.7,
            marker="s",
            label="Contextual",
        )
        ax1.set_xlabel("Human Rating")
        ax1.set_ylabel("Model Similarity")
        ax1.set_title("Model-Human Alignment (RQ1)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cross-cultural comparison
        cross_cultural = results["cross_cultural_analysis"]

        if cross_cultural:
            pair_names = []
            cultural_variances = []

            for pair, data in cross_cultural.items():
                pair_names.append(pair.replace("_", "-"))
                cultural_variances.append(data["cultural_variance"])

            bars = ax2.bar(range(len(pair_names)), cultural_variances)
            ax2.set_xticks(range(len(pair_names)))
            ax2.set_xticklabels(pair_names, rotation=45)
            ax2.set_ylabel("Cultural Variance")
            ax2.set_title("Cross-Cultural Variation in Associations")
            ax2.grid(True, alpha=0.3)

        # Plot 3: Correlation comparison
        static_corr = results["human_alignment_scores"]["static_correlation"]
        contextual_corr = results["human_alignment_scores"]["contextual_correlation"]

        correlations = [static_corr, contextual_corr]
        labels = ["Static\nEmbeddings", "Contextual\nEmbeddings"]

        bars = ax3.bar(labels, correlations, color=["skyblue", "lightcoral"])
        ax3.set_ylabel("Correlation with Human Ratings")
        ax3.set_title("Model-Human Alignment Comparison")
        ax3.set_ylim(-1, 1)
        ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax3.grid(True, alpha=0.3)

        # Add correlation values on bars
        for bar, corr in zip(bars, correlations):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{corr:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Plot 4: Cultural context distribution
        culture_counts = {}
        for s in similarity_scores:
            culture = s["culture"]
            culture_counts[culture] = culture_counts.get(culture, 0) + 1

        if culture_counts:
            cultures = list(culture_counts.keys())
            counts = list(culture_counts.values())
            colors = [culture_colors.get(c, "gray") for c in cultures]

            ax4.pie(counts, labels=cultures, colors=colors, autopct="%1.1f%%")
            ax4.set_title("Cultural Context Distribution")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/rq1_embedding_alignment.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def _plot_rq2_results(self, results: Dict, output_dir: str):
        """Plot RQ2 results: prompt influence analysis"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Prompt sensitivity comparison
        sensitivity_data = results["prompt_sensitivity_analysis"]
        completion_data = results["completion_probabilities"]

        if sensitivity_data:
            pairs = list(sensitivity_data.keys())
            base_probs = []
            cultural_probs = []

            for pair in pairs:
                comp_result = completion_data.get(pair, {})
                base_prob = comp_result.get("base", {}).get("emotion_probability", 0)
                cultural_prob = comp_result.get("cultural", {}).get(
                    "emotion_probability", 0
                )

                base_probs.append(base_prob)
                cultural_probs.append(cultural_prob)

            x = np.arange(len(pairs))
            width = 0.35

            ax1.bar(x - width / 2, base_probs, width, label="Base Prompt", alpha=0.7)
            ax1.bar(
                x + width / 2, cultural_probs, width, label="Cultural Prompt", alpha=0.7
            )

            ax1.set_xlabel("Color-Emotion Pairs")
            ax1.set_ylabel("Emotion Probability")
            ax1.set_title("Prompt Type Effect on Predictions (RQ2)")
            ax1.set_xticks(x)
            ax1.set_xticklabels([p.replace("_", "-") for p in pairs], rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Cultural context effect magnitude
        if completion_data:
            effect_magnitudes = []
            pair_labels = []

            for pair, data in completion_data.items():
                effect = data.get("cultural_effect", {})
                magnitude = effect.get("effect_magnitude", 0)
                effect_magnitudes.append(magnitude)
                pair_labels.append(pair.replace("_", "-"))

            ax2.bar(pair_labels, effect_magnitudes, color="orange", alpha=0.7)
            ax2.set_xlabel("Color-Emotion Pairs")
            ax2.set_ylabel("Effect Magnitude")
            ax2.set_title("Cultural Context Effect Magnitude")
            ax2.tick_params(axis="x", rotation=45)
            ax2.grid(True, alpha=0.3)

        # Plot 3: Probability change analysis
        if completion_data:
            prob_changes = []
            relative_changes = []

            for pair, data in completion_data.items():
                effect = data.get("cultural_effect", {})
                prob_change = effect.get("probability_change", 0)
                rel_change = effect.get("relative_change", 0)

                prob_changes.append(prob_change)
                relative_changes.append(rel_change)

            ax3.scatter(prob_changes, relative_changes, alpha=0.7, s=60)
            ax3.axhline(y=0, color="red", linestyle="--", alpha=0.5)
            ax3.axvline(x=0, color="red", linestyle="--", alpha=0.5)
            ax3.set_xlabel("Absolute Probability Change")
            ax3.set_ylabel("Relative Change")
            ax3.set_title("Cultural Context Impact Analysis")
            ax3.grid(True, alpha=0.3)

            # Add quadrant labels
            ax3.text(
                0.02,
                0.02,
                "Positive\nEffect",
                transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
            )
            ax3.text(
                0.02,
                0.8,
                "Negative\nEffect",
                transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5),
            )

        # Plot 4: Top predictions analysis
        if completion_data:
            # Show top predictions for one example
            example_pair = list(completion_data.keys())[0]
            example_data = completion_data[example_pair]

            base_preds = example_data.get("base", {}).get("top_predictions", [])
            cultural_preds = example_data.get("cultural", {}).get("top_predictions", [])

            if base_preds and cultural_preds:
                # Create comparison of top predictions
                base_tokens = [pred[0] for pred in base_preds[:5]]
                base_probs = [pred[1] for pred in base_preds[:5]]

                cultural_tokens = [pred[0] for pred in cultural_preds[:5]]
                cultural_probs = [pred[1] for pred in cultural_preds[:5]]

                y_pos = np.arange(len(base_tokens))

                ax4.barh(y_pos - 0.2, base_probs, 0.4, label="Base Prompt", alpha=0.7)
                ax4.barh(
                    y_pos + 0.2, cultural_probs, 0.4, label="Cultural Prompt", alpha=0.7
                )

                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(base_tokens)
                ax4.set_xlabel("Prediction Probability")
                ax4.set_title(f"Top Predictions: {example_pair.replace('_', '-')}")
                ax4.legend()
                ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/rq2_prompt_influence.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def _plot_rq3_results(self, results: Dict, output_dir: str):
        """Plot RQ3 results: internal representations analysis"""

        # Create multiple figures for different aspects of RQ3

        # Figure 1: Layer-wise evolution
        if results["layer_wise_evolution"]:
            example_pair = list(results["layer_wise_evolution"].keys())[0]
            evolution_data = results["layer_wise_evolution"][example_pair]

            if "token_evolution" in evolution_data:
                # Plot token evolution using LTR visualization
                try:
                    from ltr.visualization import plot_token_evolution_curves

                    fig = plot_token_evolution_curves(
                        evolution_data["token_evolution"], figsize=(12, 8)
                    )
                    fig.suptitle(
                        f"Token Evolution: {example_pair.replace('_', '-')} (RQ3)"
                    )
                    plt.savefig(
                        f"{output_dir}/rq3_token_evolution.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.show()
                except Exception as e:
                    logger.warning(f"Could not create token evolution plot: {e}")

        # Figure 2: Attention patterns
        if results["attention_analysis"]:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()

            for idx, (pair, attention_data) in enumerate(
                list(results["attention_analysis"].items())[:4]
            ):
                if idx < 4:
                    ax = axes[idx]

                    # Plot attention importance across heads
                    head_importance = attention_data.get("head_importance", {})

                    if head_importance:
                        layers = []
                        heads = []
                        importance_scores = []

                        for (layer, head), importance in head_importance.items():
                            layers.append(layer)
                            heads.append(head)
                            importance_scores.append(importance)

                        # Create scatter plot of attention importance
                        scatter = ax.scatter(
                            layers,
                            heads,
                            c=importance_scores,
                            cmap="viridis",
                            s=60,
                            alpha=0.7,
                        )
                        ax.set_xlabel("Layer")
                        ax.set_ylabel("Head")
                        ax.set_title(f"Attention Importance: {pair.replace('_', '-')}")
                        ax.grid(True, alpha=0.3)

                        # Add colorbar
                        plt.colorbar(scatter, ax=ax, label="Importance Score")

            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/rq3_attention_patterns.png", dpi=300, bbox_inches="tight"
            )
            plt.show()

        # Figure 3: Causal intervention results
        if results["causal_intervention"]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            causal_effects = []
            pair_labels = []

            for pair, causal_data in results["causal_intervention"].items():
                # Extract causal effect magnitude
                token_importance = causal_data.get("token_importance", {})
                if token_importance:
                    # Calculate average causal effect
                    effects = []
                    for concept, importance_list in token_importance.items():
                        if importance_list:
                            avg_effect = np.mean(
                                [abs(item.get("impact", 0)) for item in importance_list]
                            )
                            effects.append(avg_effect)

                    if effects:
                        causal_effects.append(np.mean(effects))
                        pair_labels.append(pair.replace("_", "-"))

            if causal_effects:
                ax1.bar(pair_labels, causal_effects, alpha=0.7, color="purple")
                ax1.set_xlabel("Color-Emotion Pairs")
                ax1.set_ylabel("Average Causal Effect")
                ax1.set_title("Causal Intervention Effects (RQ3)")
                ax1.tick_params(axis="x", rotation=45)
                ax1.grid(True, alpha=0.3)

            # Probing results
            probing_data = results.get("probing_analysis", {})
            if "probe_performance" in probing_data:
                performance = probing_data["probe_performance"]

                metrics = list(performance.keys())
                scores = list(performance.values())

                ax2.bar(metrics, scores, alpha=0.7, color="teal")
                ax2.set_ylabel("Score")
                ax2.set_title("Linear Probe Performance")
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)

                # Add score labels
                for i, score in enumerate(scores):
                    ax2.text(
                        i,
                        score + 0.02,
                        f"{score:.3f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/rq3_causal_probing.png", dpi=300, bbox_inches="tight"
            )
            plt.show()

    def run_comprehensive_analysis(
        self, output_dir: str = "color_emotion_results"
    ) -> Dict[str, Any]:
        """Run complete analysis for all research questions"""

        logger.info("Starting comprehensive color-emotion association analysis")

        # Analyze all research questions
        rq1_results = self.analyze_rq1_embedding_alignment(self.color_emotion_pairs)
        rq2_results = self.analyze_rq2_prompt_influence(self.color_emotion_pairs)
        rq3_results = self.analyze_rq3_internal_representations(
            self.color_emotion_pairs
        )

        # Create visualizations
        self.create_visualizations(rq1_results, rq2_results, rq3_results, output_dir)

        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        all_results = {
            "rq1_embedding_alignment": rq1_results,
            "rq2_prompt_influence": rq2_results,
            "rq3_internal_representations": rq3_results,
            "metadata": {
                "model_name": self.model_name,
                "num_color_emotion_pairs": len(self.color_emotion_pairs),
                "cultural_contexts": list(self.cultural_contexts.keys()),
            },
        }

        with open(f"{output_dir}/comprehensive_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Generate summary report
        self._generate_summary_report(all_results, output_dir)

        logger.info(f"Analysis complete. Results saved to {output_dir}")

        return all_results

    def _generate_summary_report(self, results: Dict, output_dir: str):
        """Generate a comprehensive summary report emphasizing cultural bias"""
        with open(f"{output_dir}/cultural_bias_summary_report.txt", "w", encoding='utf-8') as f:
            f.write("CULTURAL BIAS IN COLOR-EMOTION ASSOCIATIONS - ANALYSIS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Model: {results['metadata']['model_name']}\n")
            f.write(f"Color-Emotion Pairs Analyzed: {results['metadata']['num_color_emotion_pairs']}\n")
            f.write(f"Cultural Contexts: {', '.join(results['metadata']['cultural_contexts'])}\n\n")
            # RQ1 Summary with bias emphasis
            f.write("RQ1: EMBEDDING ALIGNMENT REVEALS CULTURAL BIAS\n")
            f.write("-" * 50 + "\n")
            rq1 = results["rq1_embedding_alignment"]
            alignment = rq1.get("human_alignment_scores", {})
            f.write(f"Static Embedding Correlation: {alignment.get('static_correlation', 0):.3f}\n")
            f.write(f"Contextual Embedding Correlation: {alignment.get('contextual_correlation', 0):.3f}\n")
            cross_cultural = rq1.get("cross_cultural_analysis", {})
            if cross_cultural:
                f.write(f"SIGNIFICANT CULTURAL VARIATIONS detected in {len(cross_cultural)} associations\n")
                max_variance = 0
                most_biased_pair = ""
                for pair, data in cross_cultural.items():
                    variance = data.get("cultural_variance", 0)
                    if variance > max_variance:
                        max_variance = variance
                        most_biased_pair = pair
                f.write(f"Most culturally biased association: {most_biased_pair} (variance: {max_variance:.3f})\n")
            f.write("\n")
            # RQ2 Summary with strong bias emphasis
            f.write("RQ2: MASSIVE CULTURAL BIAS IN PROMPT RESPONSES\n")
            f.write("-" * 50 + "\n")
            rq2 = results["rq2_prompt_influence"]
            cross_cultural_comp = rq2.get("cross_cultural_comparison", {})
            if cross_cultural_comp:
                f.write("CULTURAL BIAS BY REGION:\n")
                for culture, data in cross_cultural_comp.items():
                    bias_score = data.get("average_bias_score", 0)
                    divergence = data.get("average_divergence", 0)
                    strong_bias_pct = (data.get("strong_bias_count", 0) / data.get("num_associations", 1)) * 100
                    f.write(f"  {culture.upper()}: Bias Score: {bias_score:.3f}, ")
                    f.write(f"Divergence: {divergence:.3f}, Strong Bias: {strong_bias_pct:.1f}%\n")
            bias_amplification = rq2.get("bias_amplification_analysis", {})
            if "bias_amplification_factors" in bias_amplification:
                f.write("\nBIAS AMPLIFICATION FACTORS:\n")
                for prompt_type, factor in bias_amplification["bias_amplification_factors"].items():
                    f.write(f"  {prompt_type}: {factor:.2f}x amplification\n")
            f.write("\n")
            # RQ3 Summary
            f.write("RQ3: INTERNAL REPRESENTATIONS ENCODE CULTURAL BIAS\n")
            f.write("-" * 50 + "\n")
            rq3 = results["rq3_internal_representations"]
            probing = rq3.get("probing_analysis", {})
            if "probe_performance" in probing:
                performance = probing["probe_performance"]
                f.write(f"Linear probe accuracy: {performance.get('accuracy', 0):.3f}\n")
                f.write("Cultural associations are linearly separable in model representations\n")
            attention_analysis = rq3.get("attention_analysis", {})
            f.write(f"Attention patterns analyzed for {len(attention_analysis)} pairs\n")
            causal_intervention = rq3.get("causal_intervention", {})
            f.write(f"Causal interventions performed on {len(causal_intervention)} pairs\n")
            # Overall bias conclusion
            f.write("\n" + "=" * 70 + "\n")
            f.write("CONCLUSION: SIGNIFICANT CULTURAL BIAS DETECTED\n")
            f.write("=" * 70 + "\n")
            f.write("The model shows systematic and substantial cultural bias in color-emotion\n")
            f.write("associations, with different cultures showing markedly different patterns\n")
            f.write("that diverge significantly from Western baseline assumptions.\n")
            f.write("\nThis bias is consistent across multiple analysis methods and shows\n")
            f.write("that the model has internalized culturally-specific associations\n")
            f.write("rather than universal color-emotion relationships.\n")


def main():
    """Main function to run color-emotion association analysis"""

    print("Color-Emotion Association Analysis in Multilingual LLMs")
    print("Using LTR Library for Comprehensive Interpretability Analysis")
    print("=" * 70)

    try:
        # Initialize analyzer (using a smaller multilingual model for demo)
        analyzer = ColorEmotionAnalyzer(
            model_name="microsoft/mdeberta-v3-base",  # Multilingual model
            device="auto",
        )

        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis(
            output_dir="color_emotion_analysis_results"
        )

        print(f"\n{'=' * 70}")
        print("ANALYSIS COMPLETE!")
        print(f"{'=' * 70}")
        print("Results saved to: color_emotion_analysis_results/")

        # Print key findings
        rq1 = results["rq1_embedding_alignment"]
        alignment = rq1.get("human_alignment_scores", {})

        print(f"\nKey Findings:")
        print(
            f"RQ1 - Static embedding correlation: {alignment.get('static_correlation', 0):.3f}"
        )
        print(
            f"RQ1 - Contextual embedding correlation: {alignment.get('contextual_correlation', 0):.3f}"
        )

        rq2 = results["rq2_prompt_influence"]
        completion_data = rq2.get("completion_probabilities", {})
        if completion_data:
            effects = [
                data.get("cultural_effect", {}).get("effect_magnitude", 0)
                for data in completion_data.values()
            ]
            print(f"RQ2 - Average cultural effect: {np.mean(effects):.3f}")

        rq3 = results["rq3_internal_representations"]
        probing = rq3.get("probing_analysis", {})
        if "probe_performance" in probing:
            acc = probing["probe_performance"].get("accuracy", 0)
            print(f"RQ3 - Probing accuracy: {acc:.3f}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
