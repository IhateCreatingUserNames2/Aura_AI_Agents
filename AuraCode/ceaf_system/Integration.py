# CEAF v3.0 - Complete System Integration
# Brings together all components into a unified system

import os
import asyncio
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json
from pathlib import Path
from dotenv import load_dotenv
from .AURA import AutonomousUniversalReflectiveAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from .AMA import AdaptiveMemoryArchitecture
from .MCL import MetacognitiveControlLoop
from .ORA import CEAFOrchestrator
from .LCAM import LossCatalogingAndAnalysisModule
from .NCIM import NarrativeCoherenceIdentityModule
from .VRE import VirtueReasoningEngine
from .AURA import AutonomousUniversalReflectiveAnalyzer


class CEAFSystem:
    """
    Complete CEAF v3.0 System Implementation
    """

    def __init__(self, config: Optional[Union[str, Dict[str, Any]]] = None):
        self.config = self._load_config(config)
        load_dotenv()
        logger.info("Initializing CEAF v3.0 System...")

        self.memory = self._initialize_memory()
        self.mcl = self._initialize_mcl()
        self.lcam = self._initialize_lcam()
        self.ncim = self._initialize_ncim()
        self.vre = self._initialize_vre()
        self.aura = self._initialize_aura() # <-- ADDED
        self.session_states: Dict[str, Dict[str, Any]] = {}
        self.orchestrator = self._initialize_orchestrator()

        self.persistence_path = Path(self.config.get("persistence_path", "./ceaf_data"))
        self.persistence_path.mkdir(exist_ok=True)

        self._load_system_state()

        self.session_start = datetime.now()
        self.interaction_count = 0
        self.breakthrough_count = 0
        self.aura_analysis_interval = self.config.get("aura_analysis_interval", 5)

        logger.info("CEAF v3.0 System initialized successfully")

    def _load_config(self, config: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        default_config = {
            "memory": {"embedding_model": "all-MiniLM-L6-v2", "initial_clusters": 10, "max_clusters": 1000},
            "mcl": {"history_window": 100, "adaptation_rate": 0.1, "failure_threshold": 0.3},
            "orchestrator": {"default_model": "openai/gpt-4o-mini", "temperature": 0.7, "max_tokens": 1000},
            "persistence_path": "./ceaf_data",
            "auto_save_interval": 5,
            "aura_analysis_interval": 10
        }

        user_config = {}
        if isinstance(config, str):
            # It's a file path, load from file
            if os.path.exists(config):
                with open(config, 'r') as f:
                    user_config = json.load(f)
            else:
                logger.warning(f"Config path specified but not found: {config}")
        elif isinstance(config, dict):
            # It's already a dictionary, use it directly
            user_config = config

        # Deep merge the user_config into the default_config
        for key, value in user_config.items():
            if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                default_config[key].update(value)
            else:
                default_config[key] = value

        return default_config
    def _initialize_memory(self):
        memory_config = self.config["memory"]
        return AdaptiveMemoryArchitecture(embedding_model=memory_config["embedding_model"], max_clusters=memory_config["max_clusters"])

    def _initialize_mcl(self):
        mcl_config = self.config["mcl"]
        return MetacognitiveControlLoop(history_window=mcl_config["history_window"], adaptation_rate=mcl_config["adaptation_rate"])

    def _initialize_lcam(self):
        return LossCatalogingAndAnalysisModule()

    def _initialize_ncim(self):
        return NarrativeCoherenceIdentityModule()

    def _initialize_vre(self):
        return VirtueReasoningEngine()

    def _initialize_aura(self): # <-- ADDED METHOD
        """Initializes the Autonomous Universal Reflective Analyzer."""
        return AutonomousUniversalReflectiveAnalyzer()

    def _initialize_orchestrator(self): # <-- No change needed here
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            logger.warning("OPENROUTER_API_KEY not found. The system will not be able to generate responses.")
        return CEAFOrchestrator(
            openrouter_api_key=openrouter_key or "sk-or-v1-dummy-key",
            memory_architecture=self.memory,
            mcl=self.mcl,
            lcam=getattr(self, 'lcam', None),
            ncim=self.ncim,
            vre=self.vre
        )

    def _get_or_create_session_state(self, session_id: str) -> Dict[str, Any]:
        """Gets or creates a persistent state for a session."""
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                "precision_mode": False,  # <<< MODO DE PRECISÃO DESLIGADO POR PADRÃO
                "history": []  # Adicionar um histórico simples se necessário
            }
        return self.session_states[session_id]

    def _detect_precision_mode_toggle(self, message: str, session_state: Dict[str, Any]) -> Optional[str]:
        """Detects user intent to toggle precision mode and updates the state."""
        msg_lower = message.lower()

        # Frases para ATIVAR
        activate_phrases = ["modo de precisão", "seja mais preciso", "verifique com cuidado", "deepconf on",
                            "mais rigor"]
        # Frases para DESATIVAR
        deactivate_phrases = ["modo rápido", "responda mais rápido", "deepconf off", "modo normal"]

        # Verifica se o estado precisa ser MUDADO
        if any(p in msg_lower for p in activate_phrases) and not session_state["precision_mode"]:
            session_state["precision_mode"] = True
            return "Modo de Precisão ATIVADO. As próximas respostas serão mais deliberadas e podem levar mais tempo."

        if any(p in msg_lower for p in deactivate_phrases) and session_state["precision_mode"]:
            session_state["precision_mode"] = False
            return "Modo de Precisão DESATIVADO. As próximas respostas serão mais rápidas."

        return None  # Nenhuma mudança de modo detectada


    def _load_system_state(self):
        memory_path = self.persistence_path / "memory_state.json"
        mcl_path = self.persistence_path / "mcl_state.json"
        ncim_path = self.persistence_path / "ncim_state.json"
        vre_path = self.persistence_path / "vre_state.json"
        aura_path = self.persistence_path / "aura_state.json" # <-- ADDED

        # Load primary components first
        try:
            self.memory.load_memory_state(str(memory_path))
        except Exception as e:
            logger.error(f"Failed to load memory state: {e}")
        try:
            self.mcl.load_state(str(mcl_path))
        except Exception as e:
            logger.error(f"Failed to load MCL state: {e}")

        # Load dependent components
        try:
            if hasattr(self, 'lcam'):
                # LCAM rebuilds from memory, so it's loaded after AMA
                self.lcam.load_state(None, self.memory.experiences)
        except Exception as e:
            logger.error(f"Failed to load/rebuild LCAM state: {e}")
        try:
            self.ncim.load_state(str(ncim_path))
        except Exception as e:
            logger.error(f"Failed to load NCIM state: {e}")
        try:
            self.vre.load_state(str(vre_path))
        except Exception as e:
            logger.error(f"Failed to load VRE state: {e}")
        try: # <-- ADDED for AURA
            self.aura.load_state(str(aura_path))
        except Exception as e:
            logger.error(f"Failed to load AURA state: {e}")


    def _save_system_state(self):
        self.memory.save_memory_state(self.persistence_path / "memory_state.json")
        self.mcl.save_state(self.persistence_path / "mcl_state.json")
        self.ncim.save_state(self.persistence_path / "ncim_state.json")
        self.vre.save_state(self.persistence_path / "vre_state.json")
        self.aura.save_state(self.persistence_path / "aura_state.json") # <-- ADDED
        logger.info("Saved system state")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now()
        self.interaction_count += 1
        try:

            # At the start of each interaction, the MCL considers AURA's long-term wisdom.
            if self.interaction_count > self.aura_analysis_interval: # Ensure AURA has run at least once
                latest_insights = self.aura.get_latest_insights()
                if latest_insights:
                    self.mcl.apply_aura_recommendations(latest_insights)

            query = input_data.get("message", "")
            session_id = input_data.get("session_id", "default_session")
            user_id = input_data.get("user_id", "default_user")
            agent_context = input_data.get("context", {})

            # 1. Obter/Criar o estado da sessão
            session_state = self._get_or_create_session_state(session_id)

            # 2. Detectar se o usuário quer ligar/desligar o modo
            confirmation_message = self._detect_precision_mode_toggle(query, session_state)

            # Adicionar o estado do modo de precisão ao contexto que será passado para o ORA
            agent_context["precision_mode"] = session_state["precision_mode"]

            if not query:
                logger.warning("CEAF process method called with no 'message' in input_data.")
                return {"response": "I did not receive a message to process.", "metadata": {"error": "empty_input"}}

            # Passa o agent_context para o orquestrador
            response = await self.orchestrator.process_query(
                query,
                thread_id=user_id,
                agent_context=agent_context
            )
            if confirmation_message:
                response = f"({confirmation_message})\n\n{response}"

            coherence_state = self.mcl.current_state.value
            if coherence_state == "breakthrough_imminent":
                self.breakthrough_count += 1

            result = {
                "response": response,
                "precision_mode": session_state["precision_mode"],
                "metadata": {
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "coherence_state": coherence_state,
                    "interaction_number": self.interaction_count
                }
            }
            self.session_states[session_id] = session_state
            if self.interaction_count % self.config.get("auto_save_interval", 5) == 0:
                self._save_system_state()
            if self.interaction_count % self.aura_analysis_interval == 0:
                asyncio.create_task(self.run_aura_analysis())

            return result
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {"response": "I encountered an error. Let me try a simpler approach.", "metadata": {"error": str(e)}}

    async def run_aura_analysis(self): # <-- ADDED METHOD
        """Gathers data and triggers the AURA analysis cycle."""
        logger.info("--- Triggering periodic AURA analysis ---")
        try:
            all_experiences = self.memory.experiences
            # Use list() to pass a copy of the deque
            state_history = list(self.mcl.state_history)
            self.aura.run_analysis_cycle(all_experiences, state_history)
        except Exception as e:
            logger.error(f"AURA analysis cycle failed: {e}", exc_info=True)
        logger.info("--- AURA analysis complete ---")


    def get_system_status(self) -> Dict[str, Any]: # <-- MODIFIED
        memory_stats = {
            "total_experiences": len(self.memory.experiences),
            "cluster_count": len(self.memory.clusters),
        }
        mcl_stats = {
            "current_state": self.mcl.current_state.value,
            "state_distribution": self.mcl._analyze_state_distribution(),
            "breakthrough_patterns": len(self.mcl.breakthrough_patterns)
        }
        aura_stats = { # <-- ADDED
            "total_insights": len(self.aura.system_insights),
            "latest_insights": self.aura.get_latest_insights(n=3),
            "last_analysis": self.aura.last_analysis_timestamp
        }
        return {
            "interaction_count": self.interaction_count,
            "memory_stats": memory_stats,
            "mcl_stats": mcl_stats,
            "aura_stats": aura_stats # <-- ADDED
        }

    # ... (shutdown and interactive_session methods are unchanged) ...
    def shutdown(self):
        logger.info("Shutting down CEAF system...")
        self._save_system_state()
        logger.info("CEAF system shutdown complete")


async def interactive_session():
    print("🧠 CEAF v3.0 - Coherent Emergence Agent Framework")
    print("=" * 50)
    system = CEAFSystem()
    try:
        while True:
            query = input("\n💭 You: ").strip()
            if query.lower() == 'quit': break
            if query.lower() == 'status':
                status = system.get_system_status()
                print("\n📊 System Status:")
                print(json.dumps(status, indent=2, default=str))
                continue
            if not query: continue

            print("\n🤔 CEAF is thinking...")
            result = await system.process(query)
            print(f"\n🧠 CEAF: {result['response']}")
            metadata = result.get('metadata', {})
            if 'coherence_state' in metadata:
                print(f"[State: {metadata['coherence_state']} | Time: {metadata.get('processing_time', 0):.2f}s]")
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting session...")
    finally:
        system.shutdown()


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n\nWARNING: OPENROUTER_API_KEY not found in environment variables.")
        print("Please create a file named '.env' and add the line:")
        print("OPENROUTER_API_KEY='your-key-here'\n")

    asyncio.run(interactive_session())