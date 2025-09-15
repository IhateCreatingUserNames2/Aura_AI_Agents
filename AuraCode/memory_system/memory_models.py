# memory_system/memory_models.py - FIXED VERSION
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np


class Memory:  # This is memory_system.memory_models.Memory
    """
    A rich memory representation with multiple attributes and embeddings.
    FIXED: Standardized to use 'custom_metadata' consistently.
    """

    def __init__(self,
                 content: str,
                 memory_type: str,  # e.g., Explicit, Emotional, Procedural, etc.
                 custom_metadata: Optional[Dict[str, Any]] = None,  # FIXED: Consistently use custom_metadata
                 emotion_score: float = 0.0,  # Normalized 0-1
                 embedding: Optional[np.ndarray] = None,  # Primary embedding
                 contextual_embedding: Optional[np.ndarray] = None,  # Context-aware
                 coherence_score: float = 0.5,  # How well-structured
                 novelty_score: float = 0.5,  # How unique
                 source_document_id: Optional[str] = None,  # If from a document
                 initial_salience: float = 0.5  # Initial importance
                 ):
        self.id: str = str(uuid.uuid4())
        self.content: str = content
        self.memory_type: str = memory_type
        self.custom_metadata: Dict[str, Any] = custom_metadata or {}  # FIXED: Consistently use custom_metadata
        self.emotion_score: float = emotion_score  # Could be a dict for complex emotions
        self.creation_time: datetime = datetime.now(timezone.utc)
        self.last_accessed: datetime = self.creation_time
        self.access_count: int = 0
        self.decay_factor: float = 1.0  # Memory decay over time (1.0 = no decay)
        self.salience: float = initial_salience  # Importance/prominence of the memory

        self.coherence_score: float = coherence_score
        self.novelty_score: float = novelty_score

        self.embedding: Optional[np.ndarray] = embedding
        self.contextual_embedding: Optional[np.ndarray] = contextual_embedding

        # Connections to other memories: List of tuples (target_memory_id, strength, type_of_relation)
        self.connections: List[tuple[str, float, str]] = []
        self.source_document_id = source_document_id

        # FIXED: Ensure source is set in custom_metadata
        self.custom_metadata.setdefault('source', 'agent_interaction')  # Default source in custom_metadata

    def update_access(self):
        """Update memory access statistics and adjust salience."""
        try:
            self.last_accessed = datetime.now(timezone.utc)
            self.access_count += 1
            # Simple salience boost on access, more complex logic can be added
            self.salience = min(1.0,
                                self.salience + 0.05 * (1 / (1 + self.access_count * 0.1)))  # Diminishing returns boost
        except Exception as e:
            # Log error but don't crash
            import logging
            logging.error(f"Error updating memory access for {self.id}: {e}")

    def apply_decay(self, rate: float = 0.01, time_since_last_access_seconds: Optional[float] = None):
        """Apply memory decay based on time elapsed since last access."""
        try:
            if time_since_last_access_seconds is None:
                time_since_access = (datetime.now(timezone.utc) - self.last_accessed).total_seconds()
            else:
                time_since_access = time_since_last_access_seconds

            # More gradual decay, less aggressive than pure log for very long times
            # Decay is stronger for less salient memories
            decay_amount = rate * (np.log1p(time_since_access / 86400.0)) * (1.0 - self.salience * 0.5)
            self.decay_factor = max(0.1, self.decay_factor - decay_amount)  # Ensure decay_factor doesn't go too low
        except Exception as e:
            # Log error but don't crash
            import logging
            logging.error(f"Error applying decay to memory {self.id}: {e}")

    def get_effective_salience(self) -> float:
        """Get the effective salience considering emotion, access count, and decay."""
        try:
            # Emotional memories might decay slower or have higher base salience
            emotion_boost = 1.0 + (self.emotion_score * 0.2)  # Max 20% boost from emotion

            # Access count increases salience (diminishing returns)
            access_factor = min(1.5, 1.0 + (0.05 * np.log1p(self.access_count)))  # Max 50% boost from access

            return self.salience * emotion_boost * access_factor * self.decay_factor
        except Exception as e:
            # Log error and return base salience
            import logging
            logging.error(f"Error calculating effective salience for memory {self.id}: {e}")
            return self.salience

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary representation.
        FIXED: Consistently use 'custom_metadata' in output.
        """
        try:
            return {
                "id": self.id,
                "content": self.content,
                "memory_type": self.memory_type,
                "custom_metadata": self.custom_metadata,  # FIXED: Use custom_metadata consistently
                "emotion_score": self.emotion_score,
                "creation_time": self.creation_time.isoformat(),
                "last_accessed": self.last_accessed.isoformat(),
                "access_count": self.access_count,
                "decay_factor": self.decay_factor,
                "salience": self.salience,
                "coherence_score": self.coherence_score,
                "novelty_score": self.novelty_score,
                "connections": self.connections,
                "source_document_id": self.source_document_id,
                "embedding": self.embedding.tolist() if self.embedding is not None else None,
                "contextual_embedding": self.contextual_embedding.tolist() if self.contextual_embedding is not None else None,
            }
        except Exception as e:
            # Log error and return minimal representation
            import logging
            logging.error(f"Error converting memory {self.id} to dict: {e}")
            return {
                "id": self.id,
                "content": self.content,
                "memory_type": self.memory_type,
                "custom_metadata": self.custom_metadata,
                "error": f"Conversion error: {str(e)}"
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create memory from dictionary representation.
        FIXED: Handle both 'metadata' and 'custom_metadata' for backward compatibility.
        """
        try:
            # FIXED: Handle both old 'metadata' and new 'custom_metadata' keys for backward compatibility
            custom_metadata = data.get("custom_metadata")
            if custom_metadata is None:
                # Fallback to old 'metadata' key if custom_metadata not found
                custom_metadata = data.get("metadata", {})

            memory = cls(
                content=data["content"],
                memory_type=data["memory_type"],
                custom_metadata=custom_metadata,  # Use the resolved metadata
                emotion_score=data.get("emotion_score", 0.0),
                coherence_score=data.get("coherence_score", 0.5),
                novelty_score=data.get("novelty_score", 0.5),
                source_document_id=data.get("source_document_id"),
                initial_salience=data.get("salience", 0.5)
            )

            # Set additional fields
            memory.id = data["id"]
            memory.creation_time = datetime.fromisoformat(data["creation_time"])
            memory.last_accessed = datetime.fromisoformat(data["last_accessed"])
            memory.access_count = data.get("access_count", 0)
            memory.decay_factor = data.get("decay_factor", 1.0)
            memory.connections = data.get("connections", [])

            # Handle embeddings
            if data.get("embedding") is not None:
                memory.embedding = np.array(data["embedding"])
            if data.get("contextual_embedding") is not None:
                memory.contextual_embedding = np.array(data["contextual_embedding"])

            return memory

        except Exception as e:
            # Log error and create minimal memory
            import logging
            logging.error(f"Error creating memory from dict: {e}")
            # Create a basic memory with error info
            return cls(
                content=data.get("content", "Error loading memory"),
                memory_type=data.get("memory_type", "Error"),
                custom_metadata={"error": f"Load error: {str(e)}", "original_id": data.get("id", "unknown")}
            )

    def __repr__(self):
        try:
            return f"<Memory(id={self.id[:8]}, type='{self.memory_type}', content='{self.content[:30]}...')>"
        except Exception as e:
            return f"<Memory(error={str(e)})>"