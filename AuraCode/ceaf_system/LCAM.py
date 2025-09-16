# --- START OF FILE ceaf_system/LCAM.py ---
# CEAF v3.0 - Loss Cataloging and Analysis Module (LCAM)

import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Local import from the CEAF system
from .AMA import MemoryExperience

logger = logging.getLogger(__name__)


class LossCatalogingAndAnalysisModule:
    """
    Analyzes failure experiences to extract wisdom, identify patterns,
    and provide actionable insights for decision-making.
    ENHANCED: Uses semantic similarity instead of keyword matching.
    """

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        # --- START ENHANCEMENT ---
        # Initialize an embedding model for semantic understanding
        self.embedder = SentenceTransformer(embedding_model)
        # Cache for failure pattern embeddings to avoid re-computation
        self.pattern_embeddings: Dict[str, np.ndarray] = {}
        # --- END ENHANCEMENT ---

        # A simple archive mapping failure patterns to a list of experiences
        self.failure_archive: Dict[str, List[MemoryExperience]] = defaultdict(list)
        # A more advanced structure could map failure IDs to detailed analysis
        self.failure_analysis: Dict[str, Dict[str, Any]] = {}

        logger.info("Initialized Loss Cataloging and Analysis Module (LCAM) with semantic capabilities.")

    def catalog_failure(self, experience: MemoryExperience):
        """Adds a failure experience to the archive and caches its pattern embedding."""
        if experience.experience_type != 'failure' or not experience.failure_pattern:
            return

        # Use a unique ID for each experience if available, e.g., from metadata
        exp_id = experience.metadata.get("interaction_id", str(experience.timestamp))

        self.failure_archive[experience.failure_pattern].append(experience)

        # --- START ENHANCEMENT ---
        # Generate and cache the embedding for the failure pattern if it's new
        if experience.failure_pattern not in self.pattern_embeddings:
            # Replace underscores with spaces for more natural semantic embedding
            natural_language_pattern = experience.failure_pattern.replace("_", " ")
            pattern_embedding = self.embedder.encode(natural_language_pattern)
            self.pattern_embeddings[experience.failure_pattern] = pattern_embedding
            logger.info(f"LCAM: Generated and cached new embedding for failure pattern '{experience.failure_pattern}'")
        # --- END ENHANCEMENT ---

        # Perform a simple, initial analysis
        self.failure_analysis[exp_id] = {
            "pattern": experience.failure_pattern,
            "context_summary": experience.content[:150],
            "learning_value": experience.learning_value,
            "timestamp": experience.timestamp
        }
        logger.info(f"LCAM: Cataloged failure with pattern '{experience.failure_pattern}'")

    def get_insights_for_context(self, query: str, memory_context: List[Dict[str, Any]]) -> list[Any] | tuple[
        list[dict[str, str | bool | int]], int | float | bool]:
        """
        Analyzes the current query via semantic similarity to find relevant loss patterns.
        This is the new, enhanced implementation.
        """
        SIMILARITY_THRESHOLD = 0.65  # Tunable threshold for relevance
        insights = []

        if not self.pattern_embeddings:
            return []

        # 1. Generate an embedding for the user's query
        query_embedding = self.embedder.encode(query)

        # 2. Find semantically similar failure patterns
        candidate_patterns = []
        for pattern, pattern_embedding in self.pattern_embeddings.items():
            similarity = util.cos_sim(query_embedding, pattern_embedding).item()
            if similarity > SIMILARITY_THRESHOLD:
                candidate_patterns.append((similarity, pattern))

        # 3. Sort by relevance (similarity score) and get the top matches
        sorted_patterns = sorted(candidate_patterns, key=lambda x: x[0], reverse=True)

        # 4. Construct insights from the most relevant past failures
        for similarity, pattern in sorted_patterns[:2]:  # Limit to top 2 insights
            experiences = self.failure_archive.get(pattern)
            if not experiences:
                continue

            # Provide a summary of the most recent, relevant failure for that pattern
            latest_exp = max(experiences, key=lambda e: e.timestamp)
            insights.append({
                "insight_type": "semantic_pattern_match",
                "failure_pattern": pattern,
                "lesson": f"A past challenge semantically similar to your query was related to '{pattern}'. The key takeaway involved: {latest_exp.content[:100]}...",
                "suggested_caution": True,
                "similarity_score": round(similarity, 2)  # Add valuable metadata
            })

        highest_similarity = max([s[0] for s in sorted_patterns], default=0.0) if sorted_patterns else 0.0
        return insights, highest_similarity

    def save_state(self, filepath: str):
        # This logic remains the same. LCAM is rebuilt from AMA on load.
        logger.info(f"LCAM state is derived from AMA, skipping explicit save to {filepath}")
        pass

    def load_state(self, filepath: str, all_experiences: List[MemoryExperience]):
        """Rebuilds the failure archive and embedding cache from the main memory store (AMA)."""
        logger.info(f"LCAM: Rebuilding failure catalog and embedding cache from main memory...")
        self.failure_archive.clear()
        self.failure_analysis.clear()
        # --- START ENHANCEMENT ---
        self.pattern_embeddings.clear()
        # --- END ENHANCEMENT ---

        for exp in all_experiences:
            # The enhanced catalog_failure will now automatically rebuild the embedding cache
            self.catalog_failure(exp)

        logger.info(
            f"LCAM state rebuilt. Found {len(self.failure_analysis)} cataloged failures and {len(self.pattern_embeddings)} unique patterns.")