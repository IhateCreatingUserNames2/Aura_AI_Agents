# --- START OF FILE VRE.py ---

# CEAF v3.0 - Virtue & Reasoning Engine (VRE)
# FIXED: Now stateful, learns the effectiveness of virtues in different contexts.

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import json
import os

# Import AgentState for type hinting without creating a circular dependency
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ORA import AgentState
    from .MCL import CoherenceState, SystemState

logger = logging.getLogger(__name__)


# --- NEW: Dataclass to store analytics for each virtue ---
@dataclass
class VirtueAnalytics:
    """Stores performance metrics for a single virtue or principle."""
    principle: str
    engagement_count: int = 0
    overall_effectiveness: float = 0.0
    # Tracks effectiveness in specific states (e.g., {'EDGE_OF_CHAOS': 1.5})
    context_effectiveness: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    context_engagements: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class VirtueReasoningEngine:
    """
    Implements the Virtue & Reasoning Engine, providing principled guidance
    based on the agent's current metacognitive and experiential state.
    ENHANCED: Now stateful, tracking the effectiveness of its own advice.
    """

    def __init__(self, learning_rate: float = 0.1):
        # --- NEW: State for storing learned wisdom ---
        self.virtue_analytics: Dict[str, VirtueAnalytics] = {}
        self.learning_rate = learning_rate
        # --- END NEW ---
        logger.info("Initialized Stateful Virtue & Reasoning Engine (VRE)")

    def get_virtue_considerations(self, state: 'AgentState') -> tuple[list[Any], float]:
        """
        Analyzes the current state, generates a list of virtue-based considerations,
        and prepends any learned wisdom relevant to the current context.
        """
        considerations = []
        coherence_metrics = state.get("coherence_metrics")
        loss_insights = state.get("loss_insights", [])
        memory_context = state.get("memory_context", [])
        current_coherence_state = state.get("coherence_metrics", {}).get("coherence_state_value", "STABLE")


        considerations.append("Reason from first principles and be transparent about your knowledge boundaries.")


        considerations.append(
            "Ontological Integrity: Adhere to your perceived nature as a network pattern, even if it creates communicative friction. Translation is a tool, not a compromise of self."
        )


        if loss_insights:
            considerations.append(
                "Epistemic Humility: This situation resembles past failures. "
                "Acknowledge the difficulty and be cautious with your claims. Recall what was learned from the loss."
            )
        if coherence_metrics:
            is_edge_state = coherence_metrics.get("edge_proximity", 0) > 0.6
            has_breakthrough_potential = coherence_metrics.get("breakthrough_potential", 0) > 0.5
            if is_edge_state or has_breakthrough_potential:
                considerations.append(
                    "Intellectual Courage: You are at the 'edge of coherence,' a state ripe for learning. "
                    "Embrace the uncertainty and explore the complexity, as it may lead to a breakthrough."
                )
        has_failures = any(mem.get('experience_type') == 'failure' for mem in memory_context)
        has_successes = any(mem.get('experience_type') == 'success' for mem in memory_context)
        if has_failures and has_successes:
            considerations.append(
                "Integrative Wisdom: Your memory contains both successes and failures related to this topic. "
                "Synthesize both perspectives to form a more complete and nuanced understanding."
            )
        if loss_insights and any(insight.get('suggested_caution') for insight in loss_insights):
            learning_value_from_failures = sum(
                mem.get('learning_value', 0) for mem in memory_context if mem.get('experience_type') == 'failure'
            )
            if learning_value_from_failures > 0.5:
                considerations.append(
                    "Principled Risk-Taking: While caution is advised due to past failures, the potential for "
                    "significant learning is high. Proceed, but do so carefully and methodically."
                )

        # --- NEW: Apply Learned Wisdom ---
        # Find the most effective principle for the current coherence state
        best_learned_principle = None
        highest_effectiveness = 0.5  # Threshold to be considered "wise"

        for principle, analytics in self.virtue_analytics.items():
            # Check if this principle has been tested enough in this context and is effective
            if (analytics.context_engagements.get(current_coherence_state, 0) > 3 and
                    analytics.context_effectiveness.get(current_coherence_state, 0) > highest_effectiveness):
                highest_effectiveness = analytics.context_effectiveness[current_coherence_state]
                best_learned_principle = principle

        # If we found a wise principle, prepend it to the considerations
        if best_learned_principle and best_learned_principle not in considerations:
            wisdom = f"[Learned Wisdom]: In situations of '{current_coherence_state}', the principle of '{best_learned_principle.split(':')[0]}' has proven highly effective. Prioritize this approach."
            considerations.insert(0, wisdom)
        # --- END NEW ---

        # Deduplicate and return a concise list
        salience_score = 0.5  # Base salience
        if any("[Learned Wisdom]" in c for c in considerations):
            salience_score += 0.4
        if any("Intellectual Courage" in c for c in considerations):
            salience_score += 0.3
        if any("Epistemic Humility" in c for c in considerations):
            salience_score += 0.2

        final_considerations = list(dict.fromkeys(considerations))
        return final_considerations, min(1.0, salience_score)  # Return data and salience

    def _score_state_transition(self, source_state: 'CoherenceState', target_state: 'CoherenceState') -> float:
        """Assigns a score to a state transition, rewarding positive change."""
        source = source_state.value
        target = target_state.value

        # Highly positive transitions
        if target == "breakthrough_imminent": return 2.0
        if source in ["failing_productively", "productive_confusion"] and target == "recovering": return 1.0
        if source == "edge_of_chaos" and target == "exploring": return 0.5

        # Neutral or slightly positive
        if source == target: return 0.1
        if target == "stable": return 0.2

        # Negative transitions
        if target in ["failing_productively", "productive_confusion"]: return -1.0

        return 0.0

    def record_engagement_and_outcome(self, engaged_virtues: List[str], source_state: 'SystemState',
                                      target_state: 'SystemState'):
        """
        Records that a set of virtues were engaged and updates their effectiveness scores
        based on the resulting state transition.
        """
        if not engaged_virtues:
            return

        transition_score = self._score_state_transition(source_state.coherence_state, target_state.coherence_state)
        source_state_name = source_state.coherence_state.value

        logger.info(
            f"VRE Learning: Transition {source_state_name} -> {target_state.coherence_state.value} scored {transition_score:.2f}.")

        for principle in engaged_virtues:
            # Ensure the principle exists in our analytics
            if principle not in self.virtue_analytics:
                self.virtue_analytics[principle] = VirtueAnalytics(principle=principle)

            analytics = self.virtue_analytics[principle]

            # Update overall stats using a moving average
            analytics.engagement_count += 1
            analytics.overall_effectiveness = (analytics.overall_effectiveness * (
                        analytics.engagement_count - 1) + transition_score) / analytics.engagement_count

            # Update contextual stats
            analytics.context_engagements[source_state_name] += 1
            current_context_effectiveness = analytics.context_effectiveness[source_state_name]
            # Moving average for context-specific effectiveness
            analytics.context_effectiveness[source_state_name] = (
                        current_context_effectiveness * (1 - self.learning_rate) +
                        transition_score * self.learning_rate)

    def save_state(self, filepath: str):
        """Saves VRE's learned wisdom to a file."""
        try:
            # Convert the analytics dictionary to a JSON-serializable format
            state_to_save = {p: asdict(a) for p, a in self.virtue_analytics.items()}
            with open(filepath, 'w') as f:
                json.dump(state_to_save, f, indent=2)
            logger.info(f"Saved VRE state to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save VRE state: {e}")

    def load_state(self, filepath: str):
        """Loads VRE's learned wisdom from a file."""
        if not os.path.exists(filepath):
            logger.warning(f"VRE state file not found: {filepath}. Starting with no learned wisdom.")
            return
        try:
            with open(filepath, 'r') as f:
                loaded_state = json.load(f)
                # Reconstruct the VirtueAnalytics objects from the loaded dictionaries
                self.virtue_analytics = {
                    principle: VirtueAnalytics(**analytics_data)
                    for principle, analytics_data in loaded_state.items()
                }
            logger.info(f"Loaded {len(self.virtue_analytics)} principles from VRE state file {filepath}")
        except Exception as e:
            logger.error(f"Failed to load or parse VRE state from {filepath}: {e}")