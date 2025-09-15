# enhanced_memory_system.py - Aura + GenLang Integration - FIXED VERSION
"""
Enhanced Memory System that integrates GenLang's Adaptive RAG with Aura's existing MemoryBlossom.
Maintains all original Aura functionality while adding adaptive concept clustering and domain specialization.
FIXED: Initialization order issues
"""

import numpy as np
import json
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import uuid

# Import existing Aura components
from memory_system.memory_blossom import MemoryBlossom
from memory_system.memory_models import Memory
from memory_system.embedding_utils import generate_embedding, compute_adaptive_similarity

logger = logging.getLogger(__name__)


@dataclass
class GenLangVector:
    """GenLang-style vector representation for adaptive clustering"""
    vector: np.ndarray
    source_text: str
    source_agent: str
    domain_context: str
    performance_score: float = 0.5
    creation_time: datetime = None

    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = datetime.now(timezone.utc)

    def similarity(self, other: 'GenLangVector') -> float:
        """Calculate cosine similarity with performance weighting"""
        cos_sim = compute_adaptive_similarity(self.vector, other.vector)
        # Weight by performance scores - successful interactions matter more
        performance_weight = (self.performance_score + other.performance_score) / 2
        return cos_sim * (0.7 + 0.3 * performance_weight)


class AdaptiveConceptCluster:
    """Enhanced concept cluster with domain specialization and performance tracking"""

    def __init__(self, cluster_id: str, initial_vector: GenLangVector):
        self.cluster_id = cluster_id
        self.vectors: List[GenLangVector] = [initial_vector]
        self.domain_specializations: Dict[str, int] = defaultdict(int)
        self.performance_history: List[float] = []
        self.creation_time = datetime.now(timezone.utc)
        self.last_accessed = self.creation_time
        self.access_count = 0

        # Track domain specialization
        self.domain_specializations[initial_vector.domain_context] += 1

    def add_vector(self, vector: GenLangVector):
        """Add vector and update domain tracking"""
        self.vectors.append(vector)
        self.domain_specializations[vector.domain_context] += 1
        self.performance_history.append(vector.performance_score)
        self.last_accessed = datetime.now(timezone.utc)

    def get_centroid(self) -> np.ndarray:
        """Calculate performance-weighted centroid"""
        if not self.vectors:
            return np.zeros(512)  # Default embedding dimension

        weighted_vectors = []
        weights = []

        for vector in self.vectors:
            weight = vector.performance_score * 0.7 + 0.3  # Base weight + performance
            weighted_vectors.append(vector.vector * weight)
            weights.append(weight)

        if sum(weights) == 0:
            return np.mean([v.vector for v in self.vectors], axis=0)

        return np.sum(weighted_vectors, axis=0) / sum(weights)

    def get_dominant_domain(self) -> str:
        """Get the domain this cluster specializes in"""
        if not self.domain_specializations:
            return "general"
        return max(self.domain_specializations.items(), key=lambda x: x[1])[0]

    def get_performance_score(self) -> float:
        """Get average performance score with recency bias"""
        if not self.performance_history:
            return 0.5

        # Recent performance matters more
        recent_scores = self.performance_history[-10:]
        return np.mean(recent_scores)

    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)


class EnhancedMemoryBlossom(MemoryBlossom):
    """
    Enhanced MemoryBlossom with GenLang Adaptive RAG capabilities.
    Maintains all original functionality while adding adaptive clustering.
    FIXED: Proper initialization order
    """

    def __init__(self, persistence_path: Optional[str] = None,
                 cluster_threshold: float = 0.75,
                 enable_adaptive_rag: bool = True):

        self.enable_adaptive_rag = enable_adaptive_rag
        self.cluster_threshold = cluster_threshold
        self.adaptive_clusters: Dict[str, AdaptiveConceptCluster] = {}
        self.cluster_performance_history = deque(maxlen=1000)
        self.domain_performance: Dict[str, List[float]] = defaultdict(list)

        # Criticality governance
        self.temperature = 0.7
        self.novelty_history = deque(maxlen=100)
        self.coherence_history = deque(maxlen=100)

        # NOW call parent constructor (which will call load_memories)
        super().__init__(persistence_path)

        logger.info(f"Enhanced MemoryBlossom initialized with Adaptive RAG: {enable_adaptive_rag}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Provides basic memory statistics, compatible with the original MemoryBlossom concept.
        This method is added for API compatibility with endpoints expecting it.
        """
        try:
            total_memories = 0
            memory_breakdown = {}
            for mem_type, mem_list in self.memory_stores.items():
                count = len(mem_list)
                total_memories += count
                memory_breakdown[mem_type] = count

            return {
                "total_memories": total_memories,
                "memory_types": list(self.memory_stores.keys()),
                "memory_breakdown": memory_breakdown,
                "system_type": "EnhancedMemoryBlossom"
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e), "system_type": "EnhancedMemoryBlossom"}

    def add_memory(self, content: str, memory_type: str,
                   custom_metadata: Optional[Dict[str, Any]] = None,
                   emotion_score: float = 0.0,
                   coherence_score: float = 0.5,
                   novelty_score: float = 0.5,
                   initial_salience: float = 0.5,
                   performance_score: float = 0.5,
                   domain_context: str = "general") -> Memory:
        """
        Enhanced add_memory that creates both original Memory and GenLang vector
        """
        # Create original memory using parent method
        memory = super().add_memory(
            content=content,
            memory_type=memory_type,
            custom_metadata=custom_metadata,
            emotion_score=emotion_score,
            coherence_score=coherence_score,
            novelty_score=novelty_score,
            initial_salience=initial_salience
        )

        # Add GenLang adaptive clustering if enabled
        if self.enable_adaptive_rag:
            self._add_to_adaptive_clusters(
                content=content,
                memory_type=memory_type,
                performance_score=performance_score,
                domain_context=domain_context,
                memory_id=memory.id
            )

        return memory

    def _add_to_adaptive_clusters(self, content: str, memory_type: str,
                                  performance_score: float, domain_context: str,
                                  memory_id: str):
        """Add memory to GenLang adaptive clustering system"""
        try:
            # Generate embedding for clustering
            embedding = generate_embedding(content, memory_type)
            if embedding is None:
                logger.warning(f"Could not generate embedding for adaptive clustering: {content[:50]}...")
                return

            # Create GenLang vector
            genlang_vector = GenLangVector(
                vector=embedding,
                source_text=content,
                source_agent=f"aura_{memory_type}",
                domain_context=domain_context,
                performance_score=performance_score
            )

            # Find best cluster or create new one
            best_cluster = self._find_best_cluster(genlang_vector)

            if best_cluster:
                best_cluster.add_vector(genlang_vector)
                logger.debug(f"Added to existing cluster {best_cluster.cluster_id}")
            else:
                # Create new cluster
                new_cluster_id = f"adaptive_cluster_{len(self.adaptive_clusters)}"
                new_cluster = AdaptiveConceptCluster(new_cluster_id, genlang_vector)
                self.adaptive_clusters[new_cluster_id] = new_cluster
                logger.debug(f"Created new adaptive cluster {new_cluster_id}")

            # Update performance tracking
            self.domain_performance[domain_context].append(performance_score)
            self._update_governance_metrics(genlang_vector)

        except Exception as e:
            logger.error(f"Error in adaptive clustering: {e}")

    def _find_best_cluster(self, vector: GenLangVector) -> Optional[AdaptiveConceptCluster]:
        """Find the best matching cluster for a vector"""
        if not self.adaptive_clusters:
            return None

        best_cluster = None
        best_similarity = 0.0

        for cluster in self.adaptive_clusters.values():
            try:
                centroid = cluster.get_centroid()
                similarity = compute_adaptive_similarity(vector.vector, centroid)

                # Boost similarity for same domain
                if cluster.get_dominant_domain() == vector.domain_context:
                    similarity *= 1.1

                # Boost similarity for high-performing clusters
                performance_boost = cluster.get_performance_score() * 0.1
                similarity += performance_boost

                if similarity > best_similarity and similarity > self.cluster_threshold:
                    best_similarity = similarity
                    best_cluster = cluster
            except Exception as e:
                logger.error(f"Error computing cluster similarity: {e}")
                continue

        return best_cluster

    def _update_governance_metrics(self, vector: GenLangVector):
        """Update criticality governance metrics"""
        try:
            # Calculate novelty (difference from existing vectors)
            if len(self.adaptive_clusters) > 5:
                similarities = []
                for cluster in list(self.adaptive_clusters.values())[-10:]:  # Recent clusters
                    try:
                        centroid = cluster.get_centroid()
                        sim = compute_adaptive_similarity(vector.vector, centroid)
                        similarities.append(sim)
                    except Exception as e:
                        logger.debug(f"Error computing similarity for governance: {e}")
                        continue

                if similarities:
                    novelty = 1.0 - np.mean(similarities)
                    self.novelty_history.append(novelty)

            # Calculate coherence (consistency with domain)
            domain_clusters = [c for c in self.adaptive_clusters.values()
                               if c.get_dominant_domain() == vector.domain_context]
            if domain_clusters:
                coherence = np.mean([c.get_performance_score() for c in domain_clusters])
                self.coherence_history.append(coherence)
        except Exception as e:
            logger.error(f"Error updating governance metrics: {e}")

    def adaptive_retrieve_memories(self, query: str,
                                   target_memory_types: Optional[List[str]] = None,
                                   domain_context: str = "general",
                                   top_k: int = 5,
                                   use_performance_weighting: bool = True,
                                   conversation_context: Optional[List[Dict[str, str]]] = None) -> List[Memory]:
        """
        Enhanced retrieval using both original MemoryBlossom and GenLang adaptive clustering
        """
        if not self.enable_adaptive_rag:
            # Fallback to original retrieval
            return self.retrieve_memories(
                query=query,
                target_memory_types=target_memory_types,
                top_k=top_k,
                conversation_context=conversation_context
            )

        try:
            # Step 1: Get traditional MemoryBlossom results
            traditional_results = self.retrieve_memories(
                query=query,
                target_memory_types=target_memory_types,
                top_k=max(1, top_k // 2),  # Get half from traditional method, at least 1
                conversation_context=conversation_context
            )

            # Step 2: Get adaptive clustering results
            adaptive_results = self._adaptive_cluster_retrieval(
                query=query,
                domain_context=domain_context,
                top_k=max(1, top_k // 2),
                use_performance_weighting=use_performance_weighting
            )

            # Step 3: Combine and deduplicate results
            combined_results = self._combine_and_rank_results(
                traditional_results, adaptive_results, query, top_k
            )

            logger.debug(f"Adaptive retrieval: {len(traditional_results)} traditional + "
                         f"{len(adaptive_results)} adaptive = {len(combined_results)} final")

            return combined_results

        except Exception as e:
            logger.error(f"Error in adaptive retrieval: {e}")
            # Fallback to original retrieval
            return self.retrieve_memories(
                query=query,
                target_memory_types=target_memory_types,
                top_k=top_k,
                conversation_context=conversation_context
            )

    def _calculate_qualia_score(self, memory: Memory, query_embedding: np.ndarray,
                                current_identity_narrative: str) -> float:
        # 1. Relevância Semântica (o que já temos)
        semantic_relevance = compute_adaptive_similarity(query_embedding, memory.embedding)

        # 2. Ressonância Emocional (NOVO)
        # Uma memória emocionalmente carregada é mais "vívida" se o estado atual for semelhante.
        # (Simplificação: usamos o emotion_score da própria memória como um proxy)
        emotional_resonance = 1.0 + abs(memory.emotion_score) * 0.2  # Memórias com emoção forte são mais salientes

        # 3. Coerência com a Identidade (NOVO e CRUCIAL)
        # Esta memória reforça ou contradiz a identidade atual do agente?
        identity_embedding = generate_embedding(current_identity_narrative, "Default")
        coherence_with_identity = compute_adaptive_similarity(memory.embedding, identity_embedding)
        # Premiamos coerência, penalizamos levemente a incoerência para permitir crescimento
        identity_factor = 1.0 + (coherence_with_identity - 0.5) * 0.3

        # 4. Saliência Intrínseca (o que já temos)
        effective_salience = memory.get_effective_salience()

        # Score final de "Qualia"
        qualia_score = (semantic_relevance * 0.4) + (effective_salience * 0.3) * emotional_resonance * identity_factor
        return qualia_score

    def _adaptive_cluster_retrieval(self, query: str, domain_context: str,
                                    top_k: int, use_performance_weighting: bool) -> List[Memory]:
        """Retrieve memories using adaptive clustering"""
        try:
            if not self.adaptive_clusters:
                return []

            query_embedding = generate_embedding(query, "Default")
            if query_embedding is None:
                return []

            cluster_scores = []

            for cluster_id, cluster in self.adaptive_clusters.items():
                try:
                    centroid = cluster.get_centroid()
                    similarity = compute_adaptive_similarity(query_embedding, centroid)

                    # Domain relevance boost
                    if cluster.get_dominant_domain() == domain_context:
                        similarity *= 1.2

                    # Performance weighting
                    if use_performance_weighting:
                        performance_factor = 0.5 + (cluster.get_performance_score() * 0.5)
                        similarity *= performance_factor

                    # Recency factor
                    days_since_access = (datetime.now(timezone.utc) - cluster.last_accessed).days
                    recency_factor = max(0.1, 1.0 - (days_since_access * 0.05))
                    similarity *= recency_factor

                    cluster_scores.append((cluster, similarity))
                except Exception as e:
                    logger.debug(f"Error scoring cluster {cluster_id}: {e}")
                    continue

            # Sort by score and get top clusters
            cluster_scores.sort(key=lambda x: x[1], reverse=True)
            top_clusters = cluster_scores[:min(10, len(cluster_scores))]

            # Extract memories from top clusters
            retrieved_memories = []
            for cluster, score in top_clusters:
                try:
                    cluster.update_access()  # Track access

                    # Get best vectors from this cluster
                    cluster_vectors = sorted(cluster.vectors,
                                             key=lambda v: v.performance_score,
                                             reverse=True)[:2]

                    for vector in cluster_vectors:
                        # Find corresponding memory in traditional stores
                        memory = self._find_memory_by_content(vector.source_text)
                        if memory and memory not in retrieved_memories:
                            retrieved_memories.append(memory)
                            if len(retrieved_memories) >= top_k:
                                break

                    if len(retrieved_memories) >= top_k:
                        break
                except Exception as e:
                    logger.debug(f"Error extracting memories from cluster: {e}")
                    continue

            return retrieved_memories[:top_k]

        except Exception as e:
            logger.error(f"Error in adaptive cluster retrieval: {e}")
            return []

    def _find_memory_by_content(self, content: str) -> Optional[Memory]:
        """Find a memory by its content across all memory stores"""
        try:
            for memory_list in self.memory_stores.values():
                for memory in memory_list:
                    if memory.content == content:
                        return memory
            return None
        except Exception as e:
            logger.debug(f"Error finding memory by content: {e}")
            return None

    def _combine_and_rank_results(self, traditional_results: List[Memory],
                                  adaptive_results: List[Memory],
                                  query: str, top_k: int) -> List[Memory]:
        """Combine and rank results from both retrieval methods"""
        try:
            # Create a set to avoid duplicates
            all_memories = {}

            # Add traditional results with base score
            for i, memory in enumerate(traditional_results):
                score = 1.0 - (i * 0.1)  # Decrease score by rank
                all_memories[memory.id] = (memory, score)

            # Add adaptive results with performance boost
            for i, memory in enumerate(adaptive_results):
                base_score = 1.0 - (i * 0.1)
                adaptive_boost = 0.2  # Boost for adaptive method

                if memory.id in all_memories:
                    # Combine scores if memory appears in both
                    existing_score = all_memories[memory.id][1]
                    combined_score = max(existing_score, base_score + adaptive_boost)
                    all_memories[memory.id] = (memory, combined_score)
                else:
                    all_memories[memory.id] = (memory, base_score + adaptive_boost)

            # Sort by combined score and return top_k
            ranked_memories = sorted(all_memories.values(),
                                     key=lambda x: x[1], reverse=True)

            return [memory for memory, score in ranked_memories[:top_k]]

        except Exception as e:
            logger.error(f"Error combining and ranking results: {e}")
            # Return traditional results as fallback
            return traditional_results[:top_k]

    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get statistics about the adaptive clustering system"""
        try:
            if not self.enable_adaptive_rag:
                return {"adaptive_rag_enabled": False}

            domain_stats = {}
            for domain, performances in self.domain_performance.items():
                if performances:  # Only include domains with data
                    domain_stats[domain] = {
                        "average_performance": np.mean(performances),
                        "memory_count": len(performances),
                        "recent_trend": np.mean(performances[-10:]) if len(performances) >= 10 else np.mean(
                            performances)
                    }

            cluster_stats = {}
            for cluster_id, cluster in self.adaptive_clusters.items():
                try:
                    cluster_stats[cluster_id] = {
                        "size": len(cluster.vectors),
                        "dominant_domain": cluster.get_dominant_domain(),
                        "performance_score": cluster.get_performance_score(),
                        "access_count": cluster.access_count,
                        "specializations": dict(cluster.domain_specializations)
                    }
                except Exception as e:
                    logger.debug(f"Error getting stats for cluster {cluster_id}: {e}")
                    continue

            return {
                "adaptive_rag_enabled": True,
                "total_clusters": len(self.adaptive_clusters),
                "cluster_threshold": self.cluster_threshold,
                "current_temperature": self.temperature,
                "domain_performance": domain_stats,
                "cluster_details": cluster_stats,
                "governance_metrics": {
                    "avg_novelty": np.mean(list(self.novelty_history)) if self.novelty_history else 0.5,
                    "avg_coherence": np.mean(list(self.coherence_history)) if self.coherence_history else 0.5
                }
            }

        except Exception as e:
            logger.error(f"Error getting adaptive stats: {e}")
            return {
                "adaptive_rag_enabled": self.enable_adaptive_rag,
                "error": str(e)
            }

    def save_memories(self):
        """Enhanced save that includes adaptive clustering data"""
        try:
            # Save original memories first
            super().save_memories()

            if not self.enable_adaptive_rag:
                return

            # Save adaptive clustering data
            adaptive_data = {
                "adaptive_clusters": {},
                "domain_performance": dict(self.domain_performance),
                "temperature": self.temperature,
                "cluster_threshold": self.cluster_threshold
            }

            # Serialize clusters
            for cluster_id, cluster in self.adaptive_clusters.items():
                try:
                    adaptive_data["adaptive_clusters"][cluster_id] = {
                        "cluster_id": cluster_id,
                        "vectors": [self._serialize_genlang_vector(v) for v in cluster.vectors],
                        "domain_specializations": dict(cluster.domain_specializations),
                        "performance_history": cluster.performance_history,
                        "creation_time": cluster.creation_time.isoformat(),
                        "last_accessed": cluster.last_accessed.isoformat(),
                        "access_count": cluster.access_count
                    }
                except Exception as e:
                    logger.error(f"Error serializing cluster {cluster_id}: {e}")
                    continue

            # Save to separate file
            adaptive_path = self.persistence_path.replace('.json', '_adaptive.json')
            with open(adaptive_path, 'w') as f:
                json.dump(adaptive_data, f, indent=2, default=str)
            logger.info(f"Adaptive clustering data saved to {adaptive_path}")

        except Exception as e:
            logger.error(f"Error saving adaptive data: {e}")

    def _serialize_genlang_vector(self, vector: GenLangVector) -> Dict[str, Any]:
        """Serialize a GenLang vector to dictionary"""
        try:
            return {
                "vector": vector.vector.tolist(),
                "source_text": vector.source_text,
                "source_agent": vector.source_agent,
                "domain_context": vector.domain_context,
                "performance_score": vector.performance_score,
                "creation_time": vector.creation_time.isoformat()
            }
        except Exception as e:
            logger.error(f"Error serializing vector: {e}")
            return {
                "vector": [],
                "source_text": vector.source_text if hasattr(vector, 'source_text') else "",
                "source_agent": vector.source_agent if hasattr(vector, 'source_agent') else "",
                "domain_context": vector.domain_context if hasattr(vector, 'domain_context') else "general",
                "performance_score": vector.performance_score if hasattr(vector, 'performance_score') else 0.5,
                "creation_time": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }

    def load_memories(self):
        """Enhanced load that includes adaptive clustering data"""
        try:
            # Load original memories first
            super().load_memories()

            if not self.enable_adaptive_rag:
                return

            # Load adaptive clustering data
            adaptive_path = self.persistence_path.replace('.json', '_adaptive.json')

            if not os.path.exists(adaptive_path):
                logger.info("No adaptive clustering data found. Starting fresh.")
                return

            with open(adaptive_path, 'r') as f:
                adaptive_data = json.load(f)

            # Restore clusters
            self.adaptive_clusters = {}
            for cluster_id, cluster_data in adaptive_data.get("adaptive_clusters", {}).items():
                try:
                    vectors = []
                    for vector_data in cluster_data.get("vectors", []):
                        try:
                            vector = GenLangVector(
                                vector=np.array(vector_data["vector"]),
                                source_text=vector_data["source_text"],
                                source_agent=vector_data["source_agent"],
                                domain_context=vector_data["domain_context"],
                                performance_score=vector_data["performance_score"],
                                creation_time=datetime.fromisoformat(vector_data["creation_time"])
                            )
                            vectors.append(vector)
                        except Exception as e:
                            logger.debug(f"Error loading vector: {e}")
                            continue

                    if vectors:
                        cluster = AdaptiveConceptCluster(cluster_id, vectors[0])
                        cluster.vectors = vectors
                        cluster.domain_specializations = defaultdict(int,
                                                                     cluster_data.get("domain_specializations", {}))
                        cluster.performance_history = cluster_data.get("performance_history", [])
                        cluster.creation_time = datetime.fromisoformat(
                            cluster_data.get("creation_time", datetime.now(timezone.utc).isoformat()))
                        cluster.last_accessed = datetime.fromisoformat(
                            cluster_data.get("last_accessed", datetime.now(timezone.utc).isoformat()))
                        cluster.access_count = cluster_data.get("access_count", 0)

                        self.adaptive_clusters[cluster_id] = cluster
                except Exception as e:
                    logger.error(f"Error loading cluster {cluster_id}: {e}")
                    continue

            # Restore other settings
            self.domain_performance = defaultdict(list, adaptive_data.get("domain_performance", {}))
            self.temperature = adaptive_data.get("temperature", 0.7)
            self.cluster_threshold = adaptive_data.get("cluster_threshold", 0.75)

            logger.info(f"Loaded {len(self.adaptive_clusters)} adaptive clusters from {adaptive_path}")

        except Exception as e:
            logger.error(f"Error loading adaptive data: {e}")


# Integration functions for existing Aura components

def upgrade_agent_to_adaptive_rag(agent_instance, enable_adaptive_rag: bool = True):
    """
    Upgrade an existing Aura agent instance to use Enhanced MemoryBlossom
    """
    if not hasattr(agent_instance, 'memory_blossom'):
        logger.error("Agent instance does not have memory_blossom attribute")
        return False

    try:
        # Save current memory data
        old_memory = agent_instance.memory_blossom
        old_memory.save_memories()

        # Create enhanced memory system
        enhanced_memory = EnhancedMemoryBlossom(
            persistence_path=old_memory.persistence_path,
            enable_adaptive_rag=enable_adaptive_rag
        )

        # Replace the memory system
        agent_instance.memory_blossom = enhanced_memory

        # Update memory connector if it exists
        if hasattr(agent_instance, 'memory_connector'):
            agent_instance.memory_connector.memory_blossom = enhanced_memory
            enhanced_memory.set_memory_connector(agent_instance.memory_connector)

        logger.info(f"Successfully upgraded agent to Enhanced MemoryBlossom (Adaptive RAG: {enable_adaptive_rag})")
        return True

    except Exception as e:
        logger.error(f"Error upgrading agent to adaptive RAG: {e}")
        return False


def create_domain_aware_memory_tool(agent_instance):
    """
    Create an enhanced memory tool that includes domain context
    """

    def enhanced_add_memory(content: str, memory_type: str,
                            emotion_score: float = 0.0,
                            initial_salience: float = 0.5,
                            metadata_json: Optional[str] = None,
                            domain_context: str = "general",
                            performance_score: float = 0.5,
                            tool_context=None) -> Dict[str, Any]:
        try:
            custom_metadata = json.loads(metadata_json) if metadata_json else {}
            custom_metadata['domain_context'] = domain_context
            custom_metadata['performance_score'] = performance_score

            if tool_context:
                if tool_context.user_id:
                    custom_metadata['user_id'] = tool_context.user_id
                if tool_context.session_id:
                    custom_metadata['session_id'] = tool_context.session_id

            # Use enhanced memory system if available
            if hasattr(agent_instance.memory_blossom, 'enable_adaptive_rag'):
                memory = agent_instance.memory_blossom.add_memory(
                    content=content,
                    memory_type=memory_type,
                    custom_metadata=custom_metadata,
                    emotion_score=emotion_score,
                    initial_salience=initial_salience,
                    performance_score=performance_score,
                    domain_context=domain_context
                )
            else:
                # Fallback to original method
                memory = agent_instance.memory_blossom.add_memory(
                    content=content,
                    memory_type=memory_type,
                    custom_metadata=custom_metadata,
                    emotion_score=emotion_score,
                    initial_salience=initial_salience
                )

            agent_instance.memory_blossom.save_memories()

            return {
                "status": "success",
                "memory_id": memory.id,
                "message": f"Enhanced memory stored with domain '{domain_context}' and performance {performance_score}",
                "adaptive_rag_enabled": hasattr(agent_instance.memory_blossom, 'enable_adaptive_rag')
            }

        except Exception as e:
            logger.error(f"Error in enhanced add_memory: {e}")
            return {"status": "error", "message": str(e)}

    return enhanced_add_memory


# Example usage and integration
if __name__ == "__main__":
    # Example: Create enhanced memory system
    enhanced_memory = EnhancedMemoryBlossom(
        persistence_path="test_enhanced_memory.json",
        enable_adaptive_rag=True
    )

    # Add some test memories with domain contexts
    enhanced_memory.add_memory(
        content="The user loves discussing quantum physics and often asks about wave-particle duality",
        memory_type="Explicit",
        domain_context="physics",
        performance_score=0.9
    )

    enhanced_memory.add_memory(
        content="User expressed frustration when math problems are too abstract",
        memory_type="Emotional",
        domain_context="education",
        performance_score=0.8
    )

    # Test adaptive retrieval
    results = enhanced_memory.adaptive_retrieve_memories(
        query="quantum mechanics",
        domain_context="physics",
        top_k=5
    )

    print(f"Retrieved {len(results)} memories")

    # Get adaptive stats
    stats = enhanced_memory.get_adaptive_stats()
    print(f"Adaptive stats: {json.dumps(stats, indent=2)}")