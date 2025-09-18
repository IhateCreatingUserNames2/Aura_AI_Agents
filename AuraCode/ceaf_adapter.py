# ceaf_adapter.py - Adapter para fazer CEAF funcionar com interface NCF
# UPDATED VERSION - With dynamic token-based pricing system

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from database.models import AgentRepository, User, CreditTransaction
from agent_manager import MODEL_COSTS, MODEL_API_COSTS_USD
from ncf_processing import get_live_memory_influence_pilar4
# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from agent_manager import AgentConfig, AgentManager

# Imports CEAF (sistema a ser adaptado)
# In ceaf_adapter.py
try:
    from ceaf_system.Integration import CEAFSystem
    from ceaf_system.AMA import MemoryExperience

    CEAF_AVAILABLE = True
    print("✓ CEAF system found and loaded for ceaf_adapter.")
except ImportError as e: # <-- MODIFIED
    CEAF_AVAILABLE = False
    # --- THIS IS THE CRITICAL ADDITION ---
    print(f"❌ CEAF adapter failed to load dependencies. The specific error was: {e}")
    print("ℹ Using placeholders for CEAF adapter.")
    # --- END ADDITION ---



    class CEAFSystem:
        pass


    class MemoryExperience:
        pass

logger = logging.getLogger(__name__)


class CEAFAgentAdapter:

    def __init__(self, ceaf_system: CEAFSystem, config: 'AgentConfig', db_repo: AgentRepository):
        self.ceaf_system = ceaf_system
        self.config = config
        self.db_repo = db_repo
        self.memory_blossom = CEAFMemoryAdapter(ceaf_system)
        logger.info(f"CEAFAgentAdapter created for agent {config.agent_id}")

    def _detect_domain_context(self, message: str) -> str:
        message_lower = message.lower()
        domain_keywords = {
            'nsfw_role_play': ['nsfw', 'sexy', 'erotic', 'desire', 'pleasure', 'te excita', 'fetiche', 'fetish',
                               'role-play'],
            'creative_writing': ['story', 'write', 'creative', 'imagine', 'design', 'art', 'poem', 'character', 'plot',
                                 'narrative', 'fiction', 'novel'],
            'emotional_support': ['feel', 'sad', 'happy', 'worried', 'excited', 'frustrated', 'love', 'anxious',
                                  'depressed', 'angry', 'afraid', 'emotional', 'mood'],
            # Add other domains as needed
        }
        for domain, keywords in domain_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                logger.info(f"CEAF Adapter: Detected domain '{domain}'")
                return domain
        return "general"

    async def process_message(self, user_id: str, session_id: str, message: str,
                              session_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processes a message, handling session overrides and returning a dictionary
        to be consistent with the NCFAuraAgentInstance.
        """
        # --- START FIX: Determine model to use at the beginning ---
        model_to_use = self.config.model
        if session_overrides and 'model' in session_overrides:
            model_to_use = session_overrides['model']
            logger.info(f"CEAF Adapter: Using session override model: {model_to_use}")
        # --- END FIX ---

        # Helper for credit calculation (no changes needed)
        CREDITS_PER_DOLLAR = 500
        PROFIT_MARKUP = 3.0
        def calculate_credit_cost(model_name: str, input_tokens: int, output_tokens: int) -> int:
            prices = MODEL_API_COSTS_USD.get(model_name, MODEL_API_COSTS_USD["default"])
            input_price_per_m, output_price_per_m = prices
            cost_usd = ((input_tokens / 1_000_000) * input_price_per_m) + ((output_tokens / 1_000_000) * output_price_per_m)
            final_cost_usd = cost_usd * PROFIT_MARKUP
            credit_cost = final_cost_usd * CREDITS_PER_DOLLAR
            return max(1, int(credit_cost))

        # Credit check gatekeeper
        with self.db_repo.SessionLocal() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                logger.error(f"CEAF Adapter: Credit check failed for user {user_id}.")
                # <-- FIX: Return dictionary on error
                return {"response": "Authentication error. Could not verify your user account.", "prompt": "", "domain_context": "error"}
            if user.credits < 1:
                logger.warning(f"CEAF Adapter: User {user_id} has no credits.")
                # <-- FIX: Return dictionary on error
                return {"response": "You are out of credits! Please add more to continue.", "prompt": "", "domain_context": "billing"}

        logger.info(f"Processing message for CEAF agent '{self.config.name}': '{message[:100]}...'")

        try:
            domain_context = self._detect_domain_context(message)
            live_memory_influence = []
            if self.config.settings.get("allow_live_memory_influence", False):
                live_memory_influence = await get_live_memory_influence_pilar4(message)

            ceaf_input = {
                "user_id": user_id,
                "session_id": session_id,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "session_overrides": session_overrides,
                "domain_context": domain_context,
                "context": {
                    "agent_name": self.config.name,
                    "persona": self.config.persona,
                    "detailed_persona": self.config.detailed_persona,
                    "live_memory_influence": live_memory_influence,
                    "model_to_use": model_to_use # Pass the model to the core CEAF system
                }
            }

            # Call CEAF system
            total_input_tokens = len(f"{message} {self.config.detailed_persona}") // 4
            ceaf_response = await self.ceaf_system.process(ceaf_input)

            # Extract response text and token usage
            response_text = ceaf_response.get("response", "Erro ao processar a resposta.")
            is_precision_mode = ceaf_response.get("precision_mode", False)
            if isinstance(ceaf_response, dict):
                response_text = ceaf_response.get("response") or str(ceaf_response)
                if "usage" in ceaf_response and isinstance(ceaf_response["usage"], dict):
                    usage = ceaf_response["usage"]
                    total_input_tokens = usage.get("input_tokens", total_input_tokens)
                    total_output_tokens = usage.get("output_tokens", len(response_text) // 4)
                else:
                    total_output_tokens = len(response_text) // 4
            else:
                response_text = str(ceaf_response)
                total_output_tokens = len(response_text) // 4

            if not response_text.strip():
                response_text = f"As {self.config.name}, I couldn't generate a suitable response. Could you rephrase?"

            # Dynamic credit deduction
            final_cost = calculate_credit_cost(model_to_use, total_input_tokens, total_output_tokens)
            with self.db_repo.SessionLocal() as session:
                user_to_update = session.query(User).filter(User.id == user_id).first()
                if user_to_update:
                    if user_to_update.credits < final_cost:
                        # <-- FIX: Return dictionary on error
                        return {"response": "You do not have enough credits for that response.", "prompt": "CANCELED", "domain_context": "billing"}

                    user_to_update.credits -= final_cost
                    transaction = CreditTransaction(
                        user_id=user_id,
                        agent_id=self.config.agent_id,
                        amount=-final_cost,
                        model_used=model_to_use,
                        description=f"Chat with CEAF agent {self.config.name} ({total_input_tokens} in, {total_output_tokens} out)"
                    )
                    session.add(transaction)
                    session.commit()
                    logger.info(f"CEAF Adapter: Deducted {final_cost} credits from {user_id}. New balance: {user_to_update.credits}")

            # <-- FIX: Return the full dictionary, not just the text
            return {
                "response": response_text,
                "prompt": "CEAF_DYNAMIC_PROMPT",
                "domain_context": domain_context,
                "precision_mode": is_precision_mode
            }

        except Exception as e:
            logger.error(f"Error in CEAFAgentAdapter.process_message: {e}", exc_info=True)
            # <-- FIX: Return dictionary on error
            return {"response": f"Error processing message with CEAF agent {self.config.name}: {str(e)}", "prompt": "ERROR", "domain_context": "error"}

    # Métodos para compatibilidade com sistema NCF
    def get_memory_summary(self) -> str:
        """Compatibilidade com interface NCF"""
        try:
            # CEAF tem um sistema de memória diferente
            if hasattr(self.ceaf_system, 'get_memory_summary'):
                return self.ceaf_system.get_memory_summary()
            elif hasattr(self.ceaf_system, 'memory') and hasattr(self.ceaf_system.memory, 'get_summary'):
                return self.ceaf_system.memory.get_summary()
            else:
                return f"CEAF agent {self.config.name} memory system active"
        except Exception as e:
            logger.error(f"Error getting CEAF memory summary: {e}")
            return "Error retrieving memory summary"

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics for CEAF agent (matching NCF interface)"""
        try:
            base_stats = {
                "agent_id": self.config.agent_id,
                "agent_name": self.config.name,
                "model": self.config.model,
                "system_type": "ceaf",
                "created_at": self.config.created_at.isoformat() if self.config.created_at else None,
                "memory_system_type": "CEAF Adaptive Memory Architecture"
            }

            # Add CEAF-specific stats
            try:
                if hasattr(self.ceaf_system, 'get_system_status'):
                    ceaf_stats = self.ceaf_system.get_system_status()
                    base_stats["ceaf_stats"] = ceaf_stats

                # Add memory stats from adapter
                memory_stats = self.memory_blossom.get_memory_stats()
                base_stats["memory_stats"] = memory_stats

            except Exception as e:
                logger.warning(f"Error getting CEAF system stats: {e}")
                base_stats["stats_error"] = str(e)

            return base_stats

        except Exception as e:
            logger.error(f"Error getting CEAF enhanced stats: {e}")
            return {
                "agent_id": self.config.agent_id,
                "agent_name": self.config.name,
                "system_type": "ceaf",
                "error": str(e)
            }

    def save_state(self):
        """Salvar estado CEAF"""
        try:
            self.ceaf_system._save_system_state()
            logger.info(f"CEAF agent {self.config.agent_id} state saved successfully")
        except Exception as e:
            logger.error(f"Error saving CEAF state: {e}")


class CEAFMemoryAdapter:
    """
    Adapter que faz memória CEAF parecer com MemoryBlossom NCF

    PROBLEMA:
    - NCF usa MemoryBlossom com métodos específicos
    - CEAF usa AMA (Adaptive Memory Architecture)

    SOLUÇÃO:
    - Translate calls between the two systems
    """

    def __init__(self, ceaf_system: CEAFSystem):
        self.ceaf_system = ceaf_system
        logger.info("CEAFMemoryAdapter initialized")

    def add_memory(self, content: str, memory_type: str, emotion_score: float = 0.0,
                   initial_salience: float = 0.5, custom_metadata: Dict = None, **kwargs):
        try:
            from ceaf_system.AMA import MemoryExperience  # Local import
            # Convert NCF format to CEAF MemoryExperience
            experience = MemoryExperience(
                content=content,
                timestamp=datetime.now(),
                experience_type=memory_type.lower(),  # Map type
                context=custom_metadata or {},
                outcome_value=emotion_score,  # Map emotion to outcome
                learning_value=initial_salience,  # Map salience to learning
                metadata=custom_metadata or {}
            )
            # Use the correct attribute name: memory, not ama
            cluster_id = self.ceaf_system.memory.add_experience(experience)
            return {"status": "success", "cluster_id": cluster_id, "memory_id": str(cluster_id)}
        except Exception as e:
            logger.error(f"Error in CEAFMemoryAdapter.add_memory: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def retrieve_memories(self, query: str, top_k: int = 5, memory_types: List[str] = None, **kwargs):
        try:
            # Use the correct attribute and method name
            ceaf_results = self.ceaf_system.memory.retrieve_with_loss_context(query=query, k=top_k)

            # Convert results back to NCF-like format
            ncf_memories = []
            for idx, ceaf_exp in enumerate(ceaf_results):
                ncf_memory = {
                    "id": ceaf_exp.metadata.get("id", f"ceaf_{idx}_{ceaf_exp.timestamp}"),
                    "content": ceaf_exp.content,
                    "memory_type": ceaf_exp.experience_type,
                    "emotion_score": ceaf_exp.outcome_value,
                    "salience": ceaf_exp.learning_value,
                    "timestamp": ceaf_exp.timestamp.isoformat(),
                    "custom_metadata": ceaf_exp.metadata
                }
                ncf_memories.append(ncf_memory)
            return ncf_memories
        except Exception as e:
            logger.error(f"Error in CEAFMemoryAdapter.retrieve_memories: {e}", exc_info=True)
            return []

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Gets all experiences from the CEAF memory and returns them in a format
        compatible with NCF's Memory.to_dict(). This is crucial for publishing.
        """
        try:
            all_experiences = self.ceaf_system.memory.experiences
            ncf_formatted_memories = []
            for idx, ceaf_exp in enumerate(all_experiences):
                # This conversion logic is the same as in retrieve_memories
                ncf_memory = {
                    "id": ceaf_exp.metadata.get("id", f"ceaf_{idx}_{ceaf_exp.timestamp.isoformat()}"),
                    "content": ceaf_exp.content,
                    "memory_type": ceaf_exp.experience_type,
                    "emotion_score": ceaf_exp.outcome_value,
                    "salience": ceaf_exp.learning_value,
                    # Ensure all keys expected by the publisher are here
                    "initial_salience": ceaf_exp.learning_value,
                    "custom_metadata": ceaf_exp.metadata
                }
                ncf_formatted_memories.append(ncf_memory)
            return ncf_formatted_memories
        except Exception as e:
            logger.error(f"Error in CEAFMemoryAdapter.get_all_memories: {e}", exc_info=True)
            return []



    def get_memory_stats(self):
        """
        Mimics the statistics provided by MemoryBlossom for API compatibility.
        """
        try:
            # The CEAF memory system (AMA) stores experiences in a flat list.
            all_experiences = self.ceaf_system.memory.experiences
            total_memories = len(all_experiences)

            # We need to build the 'memory_breakdown' similar to MemoryBlossom
            memory_breakdown = {}
            for exp in all_experiences:
                # Use the original 'experience_type' from CEAF
                mem_type = exp.experience_type
                memory_breakdown[mem_type] = memory_breakdown.get(mem_type, 0) + 1

            return {
                "total_memories": total_memories,
                "memory_types": list(memory_breakdown.keys()),
                "memory_breakdown": memory_breakdown,
                "system_type": "CEAF AMA"
            }
        except Exception as e:
            logger.error(f"Error getting CEAF memory stats: {e}")
            return {"error": str(e), "system_type": "CEAF AMA"}

    def save_memories(self):
        try:
            self.ceaf_system._save_system_state()
            logger.info("CEAF state saved via NCF interface (save_memories call)")
        except Exception as e:
            logger.error(f"Error saving CEAF state via NCF interface: {e}", exc_info=True)


# FACTORY para criar instância correta
def create_agent_instance(agent_id: str, config: 'AgentConfig', db_repo: AgentRepository):
    """
    Factory que decide se criar instância NCF ou CEAF
    """
    system_type = config.settings.get("system_type", "ncf")

    if system_type == "ceaf":
        if not CEAF_AVAILABLE:
            raise RuntimeError("CEAF system not available")

        # Criar sistema CEAF
        ceaf_config = config.settings.get("ceaf_config", {})
        ceaf_system = CEAFSystem(ceaf_config)

        # Retornar ADAPTER que parece com agente NCF (COM CUSTO DINÂMICO!)
        return CEAFAgentAdapter(ceaf_system, config, db_repo)

    else:
        # Criar agente NCF normal (código existente)
        # Import here to avoid circular import
        from agent_manager import NCFAuraAgentInstance
        return NCFAuraAgentInstance(config, db_repo)
