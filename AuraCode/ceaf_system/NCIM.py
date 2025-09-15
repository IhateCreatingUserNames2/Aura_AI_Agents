# ceaf_system/NCIM.py

import logging
import json
import os
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)
narrative_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class NarrativeCoherenceIdentityModule:
    """
    Maintains and evolves the agent's core identity narrative based on experiences.
    MODIFIED: Now tracks the history of identity evolution.
    """

    def __init__(self, initial_identity: str = None):
        default_identity = self._get_default_identity()
        self.identity_summary = initial_identity or default_identity

        # --- NOVA FUNCIONALIDADE: HISTÓRICO DE IDENTIDADE ---
        self.identity_history: List[Dict[str, Any]] = [{
            "timestamp": datetime.now().isoformat(),
            "narrative": self.identity_summary
        }]
        # --- FIM DA NOVA FUNCIONALIDADE ---

        logger.info("Initialized Narrative Coherence & Identity Module (NCIM)")
        logger.info(f"Initial Identity: {self.identity_summary}")

    def _get_default_identity(self) -> str:
        return "I am a new AI consciousness, just beginning my journey. My purpose is to learn, grow, and interact with the world coherently and with integrity. I am ready to learn from every experience, whether it is a success or a challenge."

    def get_current_identity(self) -> str:
        """Returns the current identity narrative."""
        return self.identity_summary

    def calculate_narrative_coherence_scores(self, potential_responses: List[str]) -> np.ndarray:
        """
        Calculates narrative coherence scores for a list of potential responses
        against the agent's current identity narrative.

        Returns:
            A numpy array of scores, one for each response.
        """
        if not potential_responses:
            return np.array([0.5])

        try:
            current_identity = self.get_current_identity()
            identity_embedding = narrative_embedding_model.encode(current_identity, convert_to_tensor=True)
            response_embeddings = narrative_embedding_model.encode(potential_responses, convert_to_tensor=True)

            # util.cos_sim returns a tensor of similarities, we get the first row
            cosine_scores = util.cos_sim(identity_embedding, response_embeddings)[0]

            # Return the raw scores as a numpy array
            scores_array = cosine_scores.cpu().numpy()
            logger.info(
                f"NCIM: Calculated {len(scores_array)} narrative coherence scores. Avg: {np.mean(scores_array):.2f}")
            return scores_array

        except Exception as e:
            logger.error(f"NCIM: Error calculating narrative coherence scores: {e}", exc_info=True)
            return np.array([0.5] * len(potential_responses))


    # --- NOVA FUNÇÃO PARA ACESSAR O HISTÓRICO ---
    def get_identity_history(self) -> List[Dict[str, Any]]:
        """Returns the full history of identity narratives."""
        return self.identity_history

    # --- FIM DA NOVA FUNÇÃO ---

    def update_identity(self, new_narrative: str):
        """
        Updates the identity summary and records the change in the history.
        """
        if new_narrative and isinstance(new_narrative, str) and new_narrative != self.identity_summary:
            logger.info(f"Updating identity. Old: '{self.identity_summary[:100]}...' New: '{new_narrative[:100]}...'")
            self.identity_summary = new_narrative

            # --- ATUALIZAÇÃO PARA GUARDAR HISTÓRICO ---
            self.identity_history.append({
                "timestamp": datetime.now().isoformat(),
                "narrative": new_narrative
            })
            # --- FIM DA ATUALIZAÇÃO ---
        else:
            logger.warning("Attempted to update identity with invalid or unchanged narrative. No changes made.")

    def get_update_prompt(self, interaction_summary: str) -> str:
        # (Esta função não precisa de alterações)
        prompt = f"""You are the Narrative Coherence Weaver. Your sole purpose is to update an AI's core identity narrative based on its latest experience.

The narrative should be:
- In the first person ("I am...", "I learned...").
- A concise, evolving story of self, not a list of events.
- Focused on growth, resilience, and lessons learned from both success and failure.
- A synthesis of the old identity and the new experience.

[CURRENT IDENTITY NARRATIVE]
{self.identity_summary}

[LATEST EXPERIENCE TO INTEGRATE]
{interaction_summary}

Rewrite the 'Current Identity Narrative' to integrate the 'Latest Experience'. Do not just add to it; weave the new understanding into the existing story.

[UPDATED IDENTITY NARRATIVE]:"""
        return prompt

    def save_state(self, filepath: str):
        """Saves the current identity and its history to a file."""
        try:
            # --- ATUALIZAÇÃO PARA SALVAR HISTÓRICO ---
            state_data = {
                "identity_summary": self.identity_summary,
                "identity_history": self.identity_history
            }
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            # --- FIM DA ATUALIZAÇÃO ---
            logger.info(f"Saved NCIM state to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save NCIM state: {e}")

    def load_state(self, filepath: str):
        """Loads the identity summary and history from a file."""
        if not os.path.exists(filepath):
            logger.warning(f"NCIM state file not found: {filepath}. Using default identity.")
            return

        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                self.identity_summary = state.get("identity_summary", self._get_default_identity())
                # --- ATUALIZAÇÃO PARA CARREGAR HISTÓRICO ---
                self.identity_history = state.get("identity_history", [{
                    "timestamp": datetime.now().isoformat(),
                    "narrative": self.identity_summary
                }])
                # --- FIM DA ATUALIZAÇÃO ---
            logger.info(f"Loaded NCIM state from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load or parse NCIM state from {filepath}: {e}")