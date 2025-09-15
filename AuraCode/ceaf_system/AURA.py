# --- START OF FILE AURA.py ---

# CEAF v3.0 - Autonomous Universal Reflective Analyzer (AURA)
# FIXED: Recommendations are now machine-actionable for the MCL.

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
import json
from .AMA import MemoryExperience

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .MCL import SystemState, CoherenceState
logger = logging.getLogger(__name__)


@dataclass
class SystemInsight:
    """A single, high-level insight about the system's long-term behavior."""

    insight_type: str  # e.g., "Loss-to-Success-Cycle", "Capability_Emergence", "Systemic_Resilience_Shift"
    summary: str
    supporting_evidence: List[Dict[str, Any]]

    # --- ENHANCEMENT: Actionable Recommendations ---
    # Human-readable summary of the recommendation
    recommendation: Optional[str] = None
    # Machine-readable instruction for other modules (e.g., the MCL)
    actionable_recommendation: Optional[Dict[str, Any]] = None
    # --- END ENHANCEMENT ---

    insight_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)


class AutonomousUniversalReflectiveAnalyzer:
    """
    Analyzes long-term historical data to identify systemic patterns,
    evolutionary trends, and deep-seated behavioral loops.
    """

    def __init__(self):
        self.system_insights: List[SystemInsight] = []
        self.last_analysis_timestamp: Optional[datetime] = None
        logger.info("Initialized Autonomous Universal Reflective Analyzer (AURA)")

    def run_analysis_cycle(self, all_experiences: List[MemoryExperience], state_history: List['SystemState']):
        """
        The main entry point for AURA's reflective process.
        This should be called periodically, not on every interaction.
        """
        if not all_experiences or not state_history:
            logger.info("AURA: Not enough data to run analysis cycle.")
            return

        logger.info(
            f"AURA: Starting analysis cycle with {len(all_experiences)} experiences and {len(state_history)} states.")

        # Run different analysis functions
        self._analyze_loss_to_breakthrough_cycles(state_history)
        self._identify_adversity_catalyzed_growth(all_experiences)
        self._assess_systemic_resilience(state_history)

        self.last_analysis_timestamp = datetime.now()
        logger.info(f"AURA: Analysis cycle complete. {len(self.system_insights)} total insights.")

    def _analyze_loss_to_breakthrough_cycles(self, state_history: List['SystemState']):
        """
        Identifies patterns where states of failure or confusion
        directly precede a 'breakthrough_imminent' state.
        """
        for i in range(1, len(state_history)):
            prev_state = state_history[i - 1]
            curr_state = state_history[i]

            is_breakthrough = curr_state.coherence_state.value == "breakthrough_imminent"
            was_in_struggle = prev_state.coherence_state.value in ["failing_productively", "productive_confusion",
                                                                   "edge_of_chaos"]

            if is_breakthrough and was_in_struggle:
                # Avoid creating duplicate insights
                if any(e.get('breakthrough_timestamp') == curr_state.timestamp.isoformat() for s in self.system_insights
                       for e in s.supporting_evidence if 'breakthrough_timestamp' in e):
                    continue

                insight = SystemInsight(
                    insight_type="Loss-to-Breakthrough-Cycle",
                    summary=f"A state of '{prev_state.coherence_state.value}' directly led to a breakthrough, demonstrating productive struggle.",
                    supporting_evidence=[
                        {"state": "preceding_struggle", "timestamp": prev_state.timestamp.isoformat(),
                         "active_failures": prev_state.active_failures},
                        {"state": "breakthrough", "timestamp": curr_state.timestamp.isoformat(),
                         "breakthrough_potential": curr_state.metrics.breakthrough_potential}
                    ],
                    # --- ENHANCEMENT: Actionable Recommendation ---
                    recommendation="Increase tolerance for productive failure, as it leads to breakthroughs.",
                    actionable_recommendation={
                        "target_module": "MCL",
                        "action": "ADJUST_COHERENCE_TARGET",
                        "parameter": "loss_tolerance",
                        "adjustment_value": 0.05,  # Propose a small increase
                        "reasoning": "Recent history shows a strong correlation between struggle states and breakthroughs."
                    }
                    # --- END ENHANCEMENT ---
                )
                self.system_insights.append(insight)
                logger.info(f"AURA Insight: Identified a Loss-to-Breakthrough cycle.")

    def _identify_adversity_catalyzed_growth(self, all_experiences: List[MemoryExperience]):
        """
        Looks for topics that were once associated with failure but are now
        associated with success, indicating learning and growth.
        """
        failure_patterns = {}
        for exp in all_experiences:
            if exp.failure_pattern:
                if exp.failure_pattern not in failure_patterns:
                    failure_patterns[exp.failure_pattern] = {'failures': 0, 'successes': 0, 'first_fail': exp.timestamp,
                                                             'last_success': None}
                failure_patterns[exp.failure_pattern]['failures'] += 1

        for exp in all_experiences:
            if exp.experience_type == 'success':
                for pattern in failure_patterns.keys():
                    if pattern in exp.content and exp.timestamp > failure_patterns[pattern]['first_fail']:
                        failure_patterns[pattern]['successes'] += 1
                        failure_patterns[pattern]['last_success'] = exp.timestamp

        for pattern, data in failure_patterns.items():
            if data['successes'] > data['failures'] / 2 and data['successes'] > 2:
                if any(e.get('pattern') == pattern for s in self.system_insights for e in s.supporting_evidence):
                    continue

                insight = SystemInsight(
                    insight_type="Adversity-Catalyzed-Growth",
                    summary=f"The system has learned to overcome the failure pattern '{pattern}'. It initially struggled but now demonstrates success in this area.",
                    supporting_evidence=[
                        {"pattern": pattern, "failure_count": data['failures'], "success_count": data['successes']}
                    ],
                    recommendation=f"The system has developed a new capability related to '{pattern}'. This knowledge should be leveraged in future reasoning."
                    # Note: This is a good candidate for a future actionable recommendation for the AMA/ORA.
                )
                self.system_insights.append(insight)
                logger.info(f"AURA Insight: Identified growth around failure pattern '{pattern}'.")

    def _assess_systemic_resilience(self, state_history: List['SystemState']):
        """
        Calculates a simple resilience score and proposes MCL target adjustments.
        """
        if len(state_history) < 20: return

        state_counts = {s.name: 0 for s in CoherenceState}
        for state in state_history:
            state_counts[state.coherence_state.name] += 1

        total_states = len(state_history)
        struggle_states = state_counts.get("FAILING_PRODUCTIVELY", 0) + state_counts.get("PRODUCTIVE_CONFUSION", 0)
        recovery_states = state_counts.get("RECOVERING", 0)
        breakthrough_states = state_counts.get("BREAKTHROUGH_IMMINENT", 0)

        resilience_score = (recovery_states + breakthrough_states * 2) / (struggle_states + 1)

        resilience_insight = next((s for s in self.system_insights if s.insight_type == "Systemic_Resilience_Metric"),
                                  None)

        # --- ENHANCEMENT: Actionable Recommendation based on score ---
        actionable_rec = None
        text_rec = ""
        if resilience_score > 1.2:  # System is highly resilient
            text_rec = "System is highly resilient. Recommend increasing creative novelty to foster more exploration."
            actionable_rec = {
                "target_module": "MCL", "action": "ADJUST_COHERENCE_TARGET",
                "parameter": "creative_novelty", "adjustment_value": 0.05,
                "reasoning": f"High resilience score ({resilience_score:.2f}) indicates capacity for more exploration."
            }
        elif resilience_score < 0.8:  # System is brittle
            text_rec = "System appears brittle. Recommend increasing semantic coherence target to promote stability."
            actionable_rec = {
                "target_module": "MCL", "action": "ADJUST_COHERENCE_TARGET",
                "parameter": "semantic_coherence", "adjustment_value": 0.05,
                "reasoning": f"Low resilience score ({resilience_score:.2f}) suggests a need for more stability."
            }
        # --- END ENHANCEMENT ---

        if resilience_insight:
            resilience_insight.summary = f"Systemic resilience score is {resilience_score:.2f}. This reflects the ability to turn struggle into growth."
            resilience_insight.supporting_evidence[0]['score'] = resilience_score
            resilience_insight.recommendation = text_rec
            resilience_insight.actionable_recommendation = actionable_rec
            resilience_insight.timestamp = datetime.now()
        else:
            insight = SystemInsight(
                insight_type="Systemic_Resilience_Metric",
                summary=f"Systemic resilience score is {resilience_score:.2f}.",
                supporting_evidence=[{"score": resilience_score, "total_states_analyzed": total_states}],
                recommendation=text_rec,
                actionable_recommendation=actionable_rec
            )
            self.system_insights.append(insight)
        logger.info(f"AURA Insight: Systemic resilience score calculated: {resilience_score:.2f}")

    def get_latest_insights(self, n: int = 5) -> List[SystemInsight]:
        """Returns the N most recent high-level insight objects."""
        return sorted(self.system_insights, key=lambda x: x.timestamp, reverse=True)[:n]

    def save_state(self, filepath: str):
        """Saves AURA's generated insights."""
        try:
            with open(filepath, 'w') as f:
                json.dump([asdict(i) for i in self.system_insights], f, indent=2, default=str)
            logger.info(f"Saved AURA state to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save AURA state: {e}")

    def load_state(self, filepath: str):
        """Loads AURA's generated insights."""
        import os
        if not os.path.exists(filepath):
            logger.warning(f"AURA state file not found: {filepath}. Starting fresh.")
            return
        try:
            with open(filepath, 'r') as f:
                insights_data = json.load(f)
                self.system_insights = [SystemInsight(**data) for data in insights_data]
            logger.info(f"Loaded {len(self.system_insights)} insights from AURA state file {filepath}")
        except Exception as e:
            logger.error(f"Failed to load or parse AURA state from {filepath}: {e}")