# CEAF v3.0 - Core Memory Architecture
# Adaptive Memory Architecture with Loss Integration

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryExperience:
    """Represents a single memory experience with success/failure context"""
    content: str
    timestamp: datetime
    experience_type: str  # 'success', 'failure', 'boundary', 'insight'
    context: Dict[str, Any]
    outcome_value: float  # -1.0 to 1.0, where negative indicates loss/failure
    learning_value: float  # 0.0 to 1.0, how much was learned
    cluster_id: Optional[int] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)  # Don't show large embedding in prints
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Loss-specific attributes
    failure_pattern: Optional[str] = None
    recovery_path: Optional[str] = None
    boundary_discovered: Optional[str] = None
    deferred_reward_timeline: Optional[int] = None  # cycles until positive outcome


@dataclass
class MemoryCluster:
    """Self-organizing memory cluster with performance tracking"""
    cluster_id: int
    cluster_type: str  # 'success', 'failure', 'mixed', 'boundary', 'meta-failure'
    experiences: List[MemoryExperience] = field(default_factory=list)

    # Performance metrics
    access_count: int = 0
    retrieval_success_rate: float = 0.0
    learning_contribution_score: float = 0.0

    # Loss integration metrics
    failure_to_insight_ratio: float = 0.0
    resilience_score: float = 0.0
    boundary_discoveries: List[str] = field(default_factory=list)

    # Cluster relationships
    paired_clusters: List[int] = field(default_factory=list)  # Success-failure pairs
    derived_clusters: List[int] = field(default_factory=list)  # Clusters that emerged from this one


class AdaptiveMemoryArchitecture:
    """
    Core memory system implementing:
    - Autonomous concept clustering
    - Power-law resource allocation
    - Success-failure cluster pairing
    - Cross-domain loss transfer
    """

    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 initial_clusters: int = 10,
                 max_clusters: int = 1000):

        # Initialize embedding model
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        # Initialize FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Memory storage
        self.experiences: List[MemoryExperience] = []
        self.clusters: Dict[int, MemoryCluster] = {}
        self.next_cluster_id = 0

        # Configuration
        self.max_clusters = max_clusters
        self.clustering_threshold = 0.3

        # Performance tracking
        self.cluster_performance: Dict[int, Dict[str, float]] = defaultdict(dict)

        # Loss pattern catalog
        self.failure_patterns: Dict[str, List[MemoryExperience]] = defaultdict(list)
        self.loss_to_insight_paths: List[Tuple[MemoryExperience, MemoryExperience]] = []

        logger.info(f"Initialized AMA with {embedding_model}, max {max_clusters} clusters")

    def add_experience(self, experience: MemoryExperience) -> int:
        """Add a new experience to memory with clustering"""

        # Generate embedding if it doesn't exist
        if experience.embedding is None:
            experience.embedding = self.embedder.encode(experience.content)

        # Find or create appropriate cluster
        cluster_id = self._assign_to_cluster(experience)
        experience.cluster_id = cluster_id

        # Store experience
        self.experiences.append(experience)
        if experience.embedding is not None:
            self.index.add(np.array([experience.embedding]))

        # Update failure pattern catalog if applicable
        if experience.experience_type == 'failure' and experience.failure_pattern:
            self.failure_patterns[experience.failure_pattern].append(experience)

        # Track loss-to-insight paths
        if experience.experience_type == 'insight' and experience.metadata.get('derived_from_failure'):
            self._track_loss_to_insight(experience)

        logger.info(f"Added {experience.experience_type} experience to cluster {cluster_id}")
        return cluster_id

    def _assign_to_cluster(self, experience: MemoryExperience) -> int:
        """Assign experience to existing cluster or create new one"""

        if not self.clusters or self.index.ntotal == 0:
            return self._create_cluster(experience)

        # Find nearest clusters
        k = min(5, self.index.ntotal)
        if k > 0 and experience.embedding is not None:
            D, I = self.index.search(np.array([experience.embedding]), k)

            # Check if any are close enough
            for dist, idx in zip(D[0], I[0]):
                if dist < self.clustering_threshold:
                    existing_exp = self.experiences[idx]
                    if existing_exp.cluster_id is not None:
                        # Add to existing cluster
                        self.clusters[existing_exp.cluster_id].experiences.append(experience)
                        self._update_cluster_metrics(existing_exp.cluster_id)
                        return existing_exp.cluster_id

        # No suitable cluster found, create new one
        return self._create_cluster(experience)

    def _create_cluster(self, experience: MemoryExperience) -> int:
        """Create a new memory cluster"""

        if len(self.clusters) >= self.max_clusters:
            # Merge least performing clusters
            self._merge_weakest_clusters()

        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1

        # Determine cluster type based on initial experience
        cluster_type = experience.experience_type
        if experience.experience_type in ['success', 'failure']:
            cluster_type = experience.experience_type
        elif experience.experience_type == 'boundary':
            cluster_type = 'boundary'
        else:
            cluster_type = 'mixed'

        self.clusters[cluster_id] = MemoryCluster(
            cluster_id=cluster_id,
            cluster_type=cluster_type,
            experiences=[experience]
        )
        logger.info(f"Created new cluster {cluster_id} of type {cluster_type}")
        return cluster_id

    def _update_cluster_metrics(self, cluster_id: int):
        """Update performance metrics for a cluster"""
        cluster = self.clusters[cluster_id]

        # Calculate failure to insight ratio
        failures = sum(1 for e in cluster.experiences if e.experience_type == 'failure')
        insights = sum(1 for e in cluster.experiences if e.experience_type == 'insight')
        if failures > 0:
            cluster.failure_to_insight_ratio = insights / failures

        # Calculate resilience score (ability to recover from failures)
        if failures > 0:
            recoveries = sum(1 for e in cluster.experiences
                             if e.recovery_path is not None)
            cluster.resilience_score = recoveries / failures

        # Update learning contribution score
        learning_values = [e.learning_value for e in cluster.experiences]
        if learning_values:
            cluster.learning_contribution_score = np.mean(learning_values)

    def retrieve_with_loss_context(self,
                                   query: str,
                                   k: int = 5,
                                   include_failures: bool = True,
                                   failure_weight: float = 0.3) -> List[MemoryExperience]:
        """
        Retrieve memories with integrated loss/failure context
        """
        if not self.experiences or self.index.ntotal == 0:
            logger.info("No experiences in memory to retrieve from.")
            return []

        query_embedding = self.embedder.encode(query)

        # Search for similar experiences
        search_k = min(k * 3, self.index.ntotal)
        D, I = self.index.search(np.array([query_embedding]), search_k)

        # Score and rank experiences
        scored_experiences = []
        for dist, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.experiences):
                exp = self.experiences[idx]
                similarity = 1 / (1 + dist)

                score = similarity
                if exp.experience_type == 'failure':
                    if include_failures:
                        score *= (1 + failure_weight * exp.learning_value)
                    else:
                        continue

                if exp.cluster_id is not None and exp.cluster_id in self.clusters:
                    score *= (1 + self.clusters[exp.cluster_id].learning_contribution_score * 0.2)

                scored_experiences.append((score, exp))

        scored_experiences.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored_experiences[:k]]

    def _track_loss_to_insight(self, insight_experience: MemoryExperience):
        """Track connections between failures and subsequent insights"""
        failure_id = insight_experience.metadata.get('derived_from_failure_id')
        if failure_id:
            for exp in self.experiences:
                if exp.metadata.get('id') == failure_id:
                    self.loss_to_insight_paths.append((exp, insight_experience))
                    logger.info(f"Tracked loss-to-insight path from {failure_id}")
                    break

    def _merge_weakest_clusters(self):
        """Merge the two weakest performing clusters"""
        if len(self.clusters) < 2:
            return

        cluster_scores = [(cid, c.learning_contribution_score * (c.access_count + 1)) for cid, c in
                          self.clusters.items()]
        cluster_scores.sort(key=lambda x: x[1])

        weak1_id, weak2_id = cluster_scores[0][0], cluster_scores[1][0]
        weak1, weak2 = self.clusters[weak1_id], self.clusters[weak2_id]

        weak1.experiences.extend(weak2.experiences)
        for exp in weak2.experiences:
            exp.cluster_id = weak1_id

        if weak1.cluster_type != weak2.cluster_type:
            weak1.cluster_type = 'mixed'

        del self.clusters[weak2_id]
        self._update_cluster_metrics(weak1_id)
        logger.info(f"Merged cluster {weak2_id} into {weak1_id}")

    def get_failure_patterns(self) -> Dict[str, List[MemoryExperience]]:
        return dict(self.failure_patterns)

    def get_resilience_insights(self) -> List[Dict[str, Any]]:
        insights = []
        for cluster in self.clusters.values():
            if cluster.resilience_score > 0.7:
                insights.append({
                    'cluster_id': cluster.cluster_id,
                    'cluster_type': cluster.cluster_type,
                    'resilience_score': cluster.resilience_score,
                    'failure_to_insight_ratio': cluster.failure_to_insight_ratio,
                    'boundary_discoveries': cluster.boundary_discoveries,
                    'example_recoveries': [e.recovery_path for e in cluster.experiences if e.recovery_path][:3]
                })
        return insights

    def save_memory_state(self, filepath: str):
        """Save the current memory state to a file."""
        state = {
            'experiences': [
                {
                    'content': e.content, 'timestamp': e.timestamp.isoformat(),
                    'experience_type': e.experience_type, 'context': e.context,
                    'outcome_value': e.outcome_value, 'learning_value': e.learning_value,
                    'cluster_id': e.cluster_id, 'metadata': e.metadata,
                    'failure_pattern': e.failure_pattern, 'recovery_path': e.recovery_path,
                    'boundary_discovered': e.boundary_discovered,
                    'deferred_reward_timeline': e.deferred_reward_timeline
                } for e in self.experiences
            ],
            'clusters': {
                str(cid): {
                    'cluster_type': c.cluster_type, 'access_count': c.access_count,
                    'retrieval_success_rate': c.retrieval_success_rate,
                    'learning_contribution_score': c.learning_contribution_score,
                    'failure_to_insight_ratio': c.failure_to_insight_ratio,
                    'resilience_score': c.resilience_score, 'boundary_discoveries': c.boundary_discoveries,
                    'paired_clusters': c.paired_clusters, 'derived_clusters': c.derived_clusters
                } for cid, c in self.clusters.items()
            },
            'next_cluster_id': self.next_cluster_id
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved memory state to {filepath}")

    def load_memory_state(self, filepath: str):
        """Load memory state from a file."""
        if not os.path.exists(filepath):
            logger.warning(f"Memory state file not found: {filepath}. Starting fresh.")
            return

        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Clear current state
            self.experiences = []
            self.clusters = {}
            self.index = faiss.IndexFlatL2(self.embedding_dim)  # Reset FAISS index

            # Load experiences and rebuild index
            loaded_experiences = []
            for e_data in state.get('experiences', []):
                exp = MemoryExperience(
                    content=e_data['content'],
                    timestamp=datetime.fromisoformat(e_data['timestamp']),
                    experience_type=e_data['experience_type'],
                    context=e_data['context'],
                    outcome_value=e_data['outcome_value'],
                    learning_value=e_data['learning_value'],
                    cluster_id=e_data['cluster_id'],
                    metadata=e_data['metadata'],
                    failure_pattern=e_data.get('failure_pattern'),
                    recovery_path=e_data.get('recovery_path'),
                    boundary_discovered=e_data.get('boundary_discovered'),
                    deferred_reward_timeline=e_data.get('deferred_reward_timeline')
                )
                # Re-generate embedding and add to index
                exp.embedding = self.embedder.encode(exp.content)
                self.index.add(np.array([exp.embedding]))
                loaded_experiences.append(exp)

            self.experiences = loaded_experiences

            # Load clusters
            for cid_str, c_data in state.get('clusters', {}).items():
                cid = int(cid_str)
                cluster_experiences = [e for e in self.experiences if e.cluster_id == cid]
                self.clusters[cid] = MemoryCluster(
                    cluster_id=cid,
                    cluster_type=c_data['cluster_type'],
                    experiences=cluster_experiences,
                    access_count=c_data.get('access_count', 0),
                    retrieval_success_rate=c_data.get('retrieval_success_rate', 0.0),
                    learning_contribution_score=c_data.get('learning_contribution_score', 0.0),
                    failure_to_insight_ratio=c_data.get('failure_to_insight_ratio', 0.0),
                    resilience_score=c_data.get('resilience_score', 0.0),
                    boundary_discoveries=c_data.get('boundary_discoveries', []),
                    paired_clusters=c_data.get('paired_clusters', []),
                    derived_clusters=c_data.get('derived_clusters', [])
                )

            self.next_cluster_id = state.get('next_cluster_id', 0)
            logger.info(
                f"Successfully loaded memory state from {filepath}. {len(self.experiences)} experiences, {len(self.clusters)} clusters.")

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load or parse memory state from {filepath}: {e}. Starting fresh.")
            # Re-initialize to a clean state
            self.__init__(self.embedder.model_name, self.max_clusters)