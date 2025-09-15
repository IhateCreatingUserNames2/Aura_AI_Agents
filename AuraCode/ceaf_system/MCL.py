# CEAF v3.0 - Metacognitive Control Loop (MCL)
# Edge of Coherence Detection with Failure Integration

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime
import asyncio
from collections import deque
import json
import inspect
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .AURA import SystemInsight
logger = logging.getLogger(__name__)


class CoherenceState(Enum):
    """System coherence states"""
    STABLE = "stable"
    EXPLORING = "exploring"
    EDGE_OF_CHAOS = "edge_of_chaos"
    PRODUCTIVE_CONFUSION = "productive_confusion"
    FAILING_PRODUCTIVELY = "failing_productively"
    RECOVERING = "recovering"
    BREAKTHROUGH_IMMINENT = "breakthrough_imminent"


@dataclass
class CoherenceMetrics:
    """Multi-dimensional coherence assessment"""
    semantic_coherence: float
    narrative_coherence: float
    epistemic_coherence: float
    creative_novelty: float
    loss_tolerance: float
    overall_coherence: float = field(init=False)
    edge_proximity: float = field(init=False)
    breakthrough_potential: float = field(init=False)

    def __post_init__(self):
        weights = {'semantic': 0.25, 'narrative': 0.25, 'epistemic': 0.20, 'creative': 0.15, 'loss': 0.15}
        self.overall_coherence = (
                    weights['semantic'] * self.semantic_coherence + weights['narrative'] * self.narrative_coherence +
                    weights['epistemic'] * self.epistemic_coherence + weights['creative'] * self.creative_novelty +
                    weights['loss'] * self.loss_tolerance)
        self.edge_proximity = (1 - abs(self.overall_coherence - 0.7)) * self.creative_novelty
        self.breakthrough_potential = self.edge_proximity * self.loss_tolerance


@dataclass
class SystemState:
    """Current system state snapshot"""
    timestamp: datetime
    coherence_state: CoherenceState
    metrics: CoherenceMetrics
    active_failures: List[str] = field(default_factory=list)
    recovery_strategies: List[str] = field(default_factory=list)
    learning_momentum: float = 0.0
    cycles_in_state: int = 0


class MetacognitiveControlLoop:
    def __init__(self, history_window: int = 100, adaptation_rate: float = 0.1, failure_threshold: float = 0.3):
        self.history_window = history_window
        self.current_state = CoherenceState.STABLE
        self.state_history: deque = deque(maxlen=history_window)
        self.metrics_history: deque = deque(maxlen=history_window)
        self.adaptation_rate = adaptation_rate
        self.failure_threshold = failure_threshold
        self.active_failures: Dict[str, Dict[str, Any]] = {}
        self.failure_recovery_patterns: Dict[str, List[str]] = {}
        self.productive_confusion_timer: Optional[datetime] = None
        self.coherence_targets = {'semantic': 0.8, 'narrative': 0.75, 'epistemic': 0.7, 'creative': 0.6, 'loss': 0.7}
        self.breakthrough_patterns: List[Tuple[Dict, Dict]] = []
        self.applied_recommendation_ids: set[str] = set()
        logger.info("Initialized Metacognitive Control Loop")

    async def assess_coherence_from_deepconf(self, deepconf_metadata: Dict[str, Any],
                                             narrative_score: float) -> CoherenceMetrics:
        """
        Assesses system coherence based on the real output of the DeepConf node.
        This replaces the old heuristic-based assessment.
        """
        # Extract metrics from the DeepConf process metadata
        total_traces = deepconf_metadata.get("total_traces_run", 1)
        valid_traces = deepconf_metadata.get("valid_traces_for_voting", 1)
        stopped_traces = deepconf_metadata.get("early_stopped_traces", 0)

        # Avoid division by zero
        if total_traces == 0: total_traces = 1

        # --- Calculate Real Metrics from DeepConf Output ---

        # Epistemic Coherence: How successful was the agent at forming a confident consensus?
        # A high ratio of valid traces to total traces means high confidence.
        epistemic_score = (valid_traces / total_traces)

        # Creative Novelty: How much "struggle" or exploration was there?
        # A high ratio of stopped traces indicates a difficult, novel, or chaotic thought process.
        creative_score = (stopped_traces / total_traces)

        # Semantic Coherence is now a measure of how well the agent converged on an answer.
        # We can also factor in the narrative score provided by NCIM.
        semantic_score = (epistemic_score * 0.7) + (narrative_score * 0.3)

        # --- Call the existing logic with REAL data ---
        # For now, current_failures can be an empty list, as DeepConf handles this implicitly
        loss_tolerance = self._calculate_loss_tolerance(current_failures=[])

        metrics = CoherenceMetrics(
            semantic_coherence=semantic_score,
            narrative_coherence=narrative_score,
            epistemic_coherence=epistemic_score,
            creative_novelty=creative_score,
            loss_tolerance=loss_tolerance
        )
        self.metrics_history.append(metrics)

        # Determine the new state based on these real metrics
        new_state = self._determine_coherence_state(metrics)

        if new_state != self.current_state:
            await self._handle_state_transition(self.current_state, new_state, metrics)
            self.current_state = new_state

        logger.info(
            f"MCL Assessment Complete. New State: {self.current_state.value}. Metrics: [Epistemic: {epistemic_score:.2f}, Creative: {creative_score:.2f}]")

        return metrics

    def apply_aura_recommendations(self, insights: List['SystemInsight']):
        """
        Applies machine-actionable recommendations from AURA to self-tune
        the MCL's coherence targets.
        """
        for insight in insights:
            rec = insight.actionable_recommendation

            # Check if recommendation is valid, for the MCL, and not already applied
            if (rec and isinstance(rec, dict) and
                    rec.get('target_module') == 'MCL' and
                    insight.insight_id not in self.applied_recommendation_ids):

                action = rec.get('action')
                parameter = rec.get('parameter')
                value = rec.get('adjustment_value')

                if action == 'ADJUST_COHERENCE_TARGET' and parameter in self.coherence_targets and isinstance(value, (
                int, float)):
                    old_value = self.coherence_targets[parameter]
                    # Apply adjustment and clamp between 0.1 and 0.9 to maintain stability
                    new_value = max(0.1, min(0.9, old_value + value))

                    if abs(new_value - old_value) > 0.001:  # Only update if there's a real change
                        self.coherence_targets[parameter] = new_value
                        logger.info(
                            f"MCL SELF-TUNING via AURA insight '{insight.insight_type}': "
                            f"Adjusted '{parameter}' target from {old_value:.2f} to {new_value:.2f}."
                        )

                    # Mark as applied to prevent re-application
                    self.applied_recommendation_ids.add(insight.insight_id)

    def get_next_turn_parameters(self) -> Dict[str, Any]:
        """
        Returns adjusted operational parameters for the ORA based on the current system state.
        This is the "Adaptive Parameter Tuner" from the CEAF blueprint.
        """
        # Start with safe defaults
        params = {
            "temperature": 0.7,
            "confidence_window": 2048,
            "confidence_threshold": 17.0,  # Default from DeepConf paper
            "warmup_traces": 1,  # Number of parallel thoughts for warmup
            "total_budget": 1  # Total parallel thoughts to generate
        }

        # Dynamically adjust based on the agent's current mental state
        if self.current_state == CoherenceState.EDGE_OF_CHAOS or self.current_state == CoherenceState.EXPLORING:
            # If we are exploring or on the edge, be more creative and tolerant
            params["temperature"] = 0.9
            # Lower the bar for stopping, allowing more "chaotic" thought
            params["confidence_threshold"] = 16.5

        elif self.current_state == CoherenceState.STABLE:
            # If stable, be more deterministic and less tolerant of deviation
            params["temperature"] = 0.5
            # Raise the bar for stopping, demanding higher confidence
            params["confidence_threshold"] = 17.5

        elif self.current_state == CoherenceState.FAILING_PRODUCTIVELY:
            # When failing, increase the search space to find a solution
            params["temperature"] = 1.0
            params["total_budget"] = 12  # Allow more "thoughts" to find a way out
            params["confidence_threshold"] = 16.0  # Be very tolerant of low confidence

        return params

    def get_next_turn_parameters(self, precision_mode: bool = False) -> Dict[str, Any]:
        """
        Returns adjusted operational parameters for the ORA.
        NOW dynamically adjusts based on Precision Mode.
        """
        if precision_mode:
            # PARÂMETROS PARA O MODO DE PRECISÃO (LENTO E ROBUSTO)
            logger.info("MCL: Generating parameters for PRECISION MODE.")
            params = {
                "temperature": 0.7,
                "confidence_threshold": 16.8,
                "warmup_traces": 3,
                "total_budget": 8
            }
        else:
            # PARÂMETROS PARA O MODO RÁPIDO (PADRÃO)
            logger.info("MCL: Generating parameters for FAST MODE.")
            params = {
                "temperature": 0.6,
                "confidence_threshold": 17.5,
                "warmup_traces": 1,
                "total_budget": 1  # APENAS UMA CHAMADA!
            }

        if precision_mode and self.current_state == CoherenceState.FAILING_PRODUCTIVELY:
            params["total_budget"] = 12
            params["temperature"] = 1.0

        return params

    def _calculate_loss_tolerance(self, current_failures: List[str]) -> float:
        tolerance = 0.5
        if current_failures and len(current_failures) > 0:
            productive_failures = sum(1 for f in current_failures if f in self.failure_recovery_patterns)
            tolerance += 0.1 * (productive_failures / len(current_failures))
        history_list = list(self.state_history)
        recent_breakthroughs = sum(
            1 for state in history_list[-10:] if state.coherence_state == CoherenceState.BREAKTHROUGH_IMMINENT)
        tolerance += 0.05 * recent_breakthroughs
        if self.productive_confusion_timer:
            duration = (datetime.now() - self.productive_confusion_timer).seconds
            if duration < 300:
                tolerance += 0.1
            elif duration > 900:
                tolerance -= 0.1
        return max(0.1, min(0.9, tolerance))

    def _determine_coherence_state(self, metrics: CoherenceMetrics) -> CoherenceState:
        # A simple bug fix: self.current_state should be checked against the value not the object
        current_state_val = self.current_state.value
        if metrics.breakthrough_potential > 0.6: return CoherenceState.BREAKTHROUGH_IMMINENT
        if metrics.loss_tolerance > 0.7 and metrics.overall_coherence < 0.5: return CoherenceState.FAILING_PRODUCTIVELY
        if 0.6 < metrics.overall_coherence < 0.8 and metrics.creative_novelty > 0.7: return CoherenceState.EDGE_OF_CHAOS
        if metrics.epistemic_coherence < 0.5 and metrics.creative_novelty > 0.6: return CoherenceState.PRODUCTIVE_CONFUSION
        if current_state_val in ["failing_productively",
                                 "productive_confusion"] and metrics.overall_coherence > 0.7: return CoherenceState.RECOVERING
        if metrics.creative_novelty > 0.5 and metrics.overall_coherence > 0.7: return CoherenceState.EXPLORING
        return CoherenceState.STABLE

    async def _handle_state_transition(self, old_state: CoherenceState, new_state: CoherenceState,
                                       metrics: CoherenceMetrics):
        logger.info(f"State transition: {old_state.value} -> {new_state.value}")
        state_snapshot = SystemState(
            timestamp=datetime.now(), coherence_state=new_state, metrics=metrics,
            active_failures=list(self.active_failures.keys()),
            learning_momentum=self._calculate_learning_momentum()
        )
        self.state_history.append(state_snapshot)

        if new_state == CoherenceState.PRODUCTIVE_CONFUSION:
            self.productive_confusion_timer = datetime.now()
        elif new_state == CoherenceState.RECOVERING:
            self.productive_confusion_timer = None
        elif new_state == CoherenceState.BREAKTHROUGH_IMMINENT:
            history_list = list(self.state_history)
            if old_state in [CoherenceState.FAILING_PRODUCTIVELY, CoherenceState.PRODUCTIVE_CONFUSION] and len(
                    history_list) > 1:
                self.breakthrough_patterns.append(
                    (asdict(history_list[-2]), asdict(state_snapshot))
                )

    def _calculate_learning_momentum(self) -> float:
        metrics_list = list(self.metrics_history)
        if len(metrics_list) < 10: return 0.0
        recent = metrics_list[-5:]
        older = metrics_list[-10:-5]
        if not older: return 0.0
        recent_avg = np.mean([m.creative_novelty * m.loss_tolerance for m in recent])
        older_avg = np.mean([m.creative_novelty * m.loss_tolerance for m in older])
        return max(-1.0, min(1.0, (recent_avg - older_avg) / max(older_avg, 0.01)))

    def _analyze_state_distribution(self) -> Dict[str, float]:
        if not self.state_history: return {CoherenceState.STABLE.value: 1.0}
        state_counts = {}
        for state in self.state_history:
            s_val = state.coherence_state.value
            state_counts[s_val] = state_counts.get(s_val, 0) + 1
        total = len(self.state_history)
        return {state_val: count / total for state_val, count in state_counts.items()}

    def register_failure(self, failure_id: str, failure_context: Dict[str, Any]):
        self.active_failures[failure_id] = {'timestamp': datetime.now(), 'context': failure_context,
                                            'recovery_attempts': 0, 'productive': False}
        logger.info(f"Registered failure: {failure_id}")

    def save_state(self, filepath: str):
        class MCLEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, datetime): return o.isoformat()
                if isinstance(o, CoherenceState): return o.value
                if isinstance(o, (CoherenceMetrics, SystemState)): return asdict(o)
                if isinstance(o, np.float32): return float(o)
                return super().default(o)

        state = {"current_state": self.current_state.value, "state_history": list(self.state_history),
                 "metrics_history": [asdict(m) for m in self.metrics_history], "active_failures": self.active_failures,
                 "failure_recovery_patterns": self.failure_recovery_patterns,
                 "productive_confusion_timer": self.productive_confusion_timer.isoformat() if self.productive_confusion_timer else None,
                 "coherence_targets": self.coherence_targets, "breakthrough_patterns": self.breakthrough_patterns, }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, cls=MCLEncoder)
        logger.info(f"Saved MCL state to {filepath}")

    def load_state(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.current_state = CoherenceState(state.get("current_state", "stable"))

            # --- MAJOR FIX: Filter dict before creating CoherenceMetrics ---
            init_params = inspect.signature(CoherenceMetrics).parameters
            metrics_init_keys = {k for k, v in init_params.items() if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD}

            loaded_history = []
            for s_dict in state.get("state_history", []):
                s_dict['timestamp'] = datetime.fromisoformat(s_dict['timestamp'])
                s_dict['coherence_state'] = CoherenceState(s_dict['coherence_state'])
                metrics_data = {k: v for k, v in s_dict['metrics'].items() if k in metrics_init_keys}
                s_dict['metrics'] = CoherenceMetrics(**metrics_data)
                loaded_history.append(SystemState(**s_dict))
            self.state_history = deque(loaded_history, maxlen=self.history_window)

            loaded_metrics = []
            for m_dict in state.get("metrics_history", []):
                # Also filter here
                metrics_data = {k: v for k, v in m_dict.items() if k in metrics_init_keys}
                loaded_metrics.append(CoherenceMetrics(**metrics_data))
            self.metrics_history = deque(loaded_metrics, maxlen=self.history_window)

            self.active_failures = state.get("active_failures", {})
            self.failure_recovery_patterns = state.get("failure_recovery_patterns", {})
            timer_str = state.get("productive_confusion_timer")
            self.productive_confusion_timer = datetime.fromisoformat(timer_str) if timer_str else None
            self.coherence_targets = state.get("coherence_targets", self.coherence_targets)
            self.breakthrough_patterns = state.get("breakthrough_patterns", [])

            logger.info(f"Loaded MCL state from {filepath}")
        except Exception as e:
            logger.warning(f"Could not load MCL state from {filepath}: {e}. Starting fresh.")