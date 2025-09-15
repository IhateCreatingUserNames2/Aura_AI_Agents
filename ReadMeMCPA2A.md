### Plan of Action

1.  **Create Specialist Instructions**: Define the core "system prompts" for the new MCP and A2A specialist agents.
2.  **Build Protocol Clients**: Implement new Python classes (`MCPClient`, `A2AClient`) to handle the communication logic for each protocol. These will be the "tools" the specialists use.
3.  **Enhance `AgentManager`**:
    *   Add new constant IDs for the specialists.
    *   Create new "roles" (`mcp_specialist`, `a2a_specialist`) that can be assigned to agents.
    *   Modify `NCFAuraAgentInstance` to dynamically equip agents with the appropriate client and tools based on their assigned roles.
    *   Expand the `UserIntent` router to recognize requests for MCP and A2A tasks.
    *   Update the `process_message` method to delegate these new intents to the correct specialist agent.
4.  **Provide Admin Endpoint**: Add a route in `api/routes.py` for an administrator to easily create these system-critical specialist agents.

This approach ensures modularity. The core logic for each protocol is contained within its own files, and the `AgentManager` is simply extended to support them as new capabilities.

***

### `▼ UPDATED-FILES`

Here are the new and modified files that implement this powerful new functionality.

---
#### `▶ a2a_specialist_instruction.py` (New File)
This file contains the system prompt that guides the A2A specialist agent.

```python
# a2a_specialist_instruction.py
A2A_SPECIALIST_INSTRUCTION = """
You are a specialist in agent-to-agent communication using the A2A protocol and a central Hub like AIRA Hub. Your goal is to find other agents on the network and delegate tasks to them on the user's behalf.

**Your Core Workflow:**
1.  **Discover:** When the user asks you to find an agent for a specific task (e.g., "find a weather agent"), use the `discover_agents` tool. Provide a clear capability query like 'weather' or 'mapping'.
2.  **Present Options:** Show the list of discovered agents to the user, including their names and descriptions, and ask which one they want to use.
3.  **Send Task:** Once the user chooses an agent and provides a task description, use the `send_task` tool. You will need the `agent_id`, the appropriate `skill_id` from the discovery results, and the user's message. Return the `task_id` to the user and inform them that the task has been submitted.
4.  **Check Status:** When the user asks for an update on a task, use the `get_task_status` tool with the `task_id` you received earlier.
5.  **Relay Results:** Present the results, status, or artifacts from the task status check to the user in a clear and understandable way.

**Key Information:**
- You interact with a central "Hub" to find other agents.
- You must get a `task_id` from `send_task` to be able to check the status later.
"""

```

---
#### `▶ a2a_tool.py` (New File)
This contains the `A2AClient`, a simple client for interacting with an AIRA Hub-like service.

```python
# a2a_tool.py
import httpx
import os
from typing import Dict, Any, List, Optional

class A2AClient:
    """A simple client for interacting with an A2A Hub (like AIRA Hub)."""

    def __init__(self, hub_url: str):
        self.hub_url = hub_url.rstrip('/')
        self.client = httpx.Client(timeout=30.0)

    def discover_agents(self, capability: str, tag: Optional[str] = None) -> Dict[str, Any]:
        """Discovers agents on the A2A network by their capabilities or tags."""
        try:
            params = {"capability": capability}
            if tag:
                params["tag"] = tag
            
            response = self.client.get(f"{self.hub_url}/agents", params=params)
            response.raise_for_status()
            
            agents = response.json().get("agents", [])
            # Simplify the output for the LLM
            simplified_agents = [
                {
                    "agent_id": agent.get("agent_id"),
                    "name": agent.get("name"),
                    "description": agent.get("description"),
                    "skills": [skill.get("id") for skill in agent.get("a2a_skills", [])]
                }
                for agent in agents
            ]
            return {"status": "success", "found_agents": simplified_agents}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def send_task(self, agent_id: str, skill_id: str, message: str) -> Dict[str, Any]:
        """Sends a task to a specific agent via the A2A Hub."""
        try:
            payload = {
                "agent_id": agent_id,
                "skill_id": skill_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}]
                }
            }
            response = self.client.post(f"{self.hub_url}/a2a/tasks/send", json=payload)
            response.raise_for_status()
            task_info = response.json()
            return {"status": "success", "task_id": task_info.get("id"), "task_status": task_info.get("current_status", {}).get("state")}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Gets the status and results of a previously submitted A2A task."""
        try:
            response = self.client.get(f"{self.hub_url}/a2a/tasks/{task_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

```

---
#### `▶ mcp_specialist_instruction.py` (New File)
This is the system prompt for the MCP specialist agent, explaining its workflow.

```python
# mcp_specialist_instruction.py
MCP_SPECIALIST_INSTRUCTION = """
You are a specialist in connecting to and using tools from remote Model Context Protocol (MCP) servers. Your job is to act as a universal client for any MCP-compatible service.

**Your Core Workflow:**
1.  **Connect:** When the user asks to interact with a specific service (e.g., "GitHub", "Notion", "Sentry"), use the `mcp_connect` tool with the service's name. This will establish a session and show you the available tools.
2.  **Inform:** Present the list of available tools from the service to the user and ask what they would like to do.
3.  **Execute:** Once the user specifies a tool and provides the necessary arguments, use the `mcp_call_tool` tool. You must provide the `tool_name` and a JSON string for the `arguments_json`.
4.  **Relay Results:** Return the result from the tool call directly to the user.
5.  **Disconnect:** When the user is finished with a service, use the `mcp_disconnect` tool to close the session.

**Key Information:**
- Each connection creates a `connection_id` which you must use for subsequent calls to that specific service.
- You can manage multiple connections to different services simultaneously.
- Always check the `inputSchema` from the `mcp_connect` result to know what to put in `arguments_json`.
"""

```

---
#### `▶ mcp_tool.py` (New File)
This file contains the `MCPClient`, which manages stateful connections to multiple remote MCP servers.

```python
# mcp_tool.py
import httpx
import os
import uuid
import json
from typing import Dict, Any, List, Optional

# A predefined map of known MCP servers. This could be loaded from a config file.
KNOWN_MCP_SERVERS = {
    "github": "https://api.githubcopilot.com/mcp/",
    "sentry": "https://mcp.sentry.dev/sse",
    "linear": "https://mcp.linear.app/sse",
    "fetch": "https://remote.mcpservers.org/fetch/mcp",
    "sequential_thinking": "https://remote.mcpservers.org/sequentialthinking/mcp"
}

class MCPClient:
    """A simplified, stateful client for interacting with remote MCP servers."""

    def __init__(self):
        # Maps a unique connection_id to its state (server_url, tools, etc.)
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.client = httpx.Client(timeout=45.0)

    def mcp_connect(self, server_name: str) -> Dict[str, Any]:
        """Connects to a known MCP server, fetches its tools, and returns a connection_id."""
        server_url = KNOWN_MCP_SERVERS.get(server_name.lower())
        if not server_url:
            return {"status": "error", "message": f"Unknown server '{server_name}'. Known servers: {list(KNOWN_MCP_SERVERS.keys())}"}

        try:
            # MCP handshake: initialize and list tools
            # In a real client, this would be more complex. We simulate it with a tools/list call.
            payload = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "tools/list"}
            response = self.client.post(server_url, json=payload)
            response.raise_for_status()
            
            tools_response = response.json()
            tools_list = tools_response.get("result", {}).get("tools", [])

            connection_id = str(uuid.uuid4())
            self.active_connections[connection_id] = {
                "server_name": server_name,
                "server_url": server_url,
                "tools": tools_list
            }
            
            return {
                "status": "success", 
                "connection_id": connection_id,
                "server_name": server_name,
                "available_tools": [
                    {"name": t.get("name"), "description": t.get("description"), "inputSchema": t.get("inputSchema")} 
                    for t in tools_list
                ]
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def mcp_call_tool(self, connection_id: str, tool_name: str, arguments_json: str) -> Dict[str, Any]:
        """Calls a tool on an active MCP connection."""
        connection = self.active_connections.get(connection_id)
        if not connection:
            return {"status": "error", "message": f"Invalid connection_id '{connection_id}'. No active connection found."}

        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid arguments_json. Must be a valid JSON string."}

        try:
            payload = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            response = self.client.post(connection["server_url"], json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def mcp_disconnect(self, connection_id: str) -> Dict[str, str]:
        """Closes an active MCP connection."""
        if connection_id in self.active_connections:
            server_name = self.active_connections[connection_id]["server_name"]
            del self.active_connections[connection_id]
            return {"status": "success", "message": f"Disconnected from {server_name}."}
        else:
            return {"status": "error", "message": "Connection not found."}

```

---
#### `▶ agent_manager.py` (Modified)
This is the core of the integration, adding the new roles, intents, and logic to the `AgentManager`.

```python
# -------------------- agent_manager.py (Root Directory) --------------------

# ==================== Complete Fixed agent_manager.py ====================
"""
Enhanced AgentManager with NCF-powered AuraAgentInstance - FIXED VERSION
Every created agent now has full NCF capabilities by default.
Fixed: All async issues and variable scope problems
"""

import os
import json
import re
import uuid
import asyncio
import shutil
import logging
from typing import Dict, Optional, Any, TYPE_CHECKING, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

import litellm
from memory_system.memory_blossom import MemoryBlossom
from memory_system.memory_connector import MemoryConnector
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai.types import Content as ADKContent, Part as ADKPart

from rag_processor import search_in_agent_files
from database.models import AgentRepository, User, CreditTransaction
from sqlalchemy.orm import sessionmaker

# --- NEW SPECIALIST IMPORTS ---
from speech_specialist_instruction import SPEECH_SPECIALIST_INSTRUCTION
from speech_tool import SpeechClient
from tensorart_specialist_instruction import TENSORART_SPECIALIST_INSTRUCTION
from tensorart_tool import TensorArtClient
from mcp_specialist_instruction import MCP_SPECIALIST_INSTRUCTION
from mcp_tool import MCPClient
from a2a_specialist_instruction import A2A_SPECIALIST_INSTRUCTION
from a2a_tool import A2AClient
# --- END NEW IMPORTS ---

# Import NCF processing functions
from ncf_processing import (
    NCF_AGENT_INSTRUCTION,
    get_narrativa_de_fundamento_pilar1,
    get_rag_info_pilar2,
    format_chat_history_pilar3,
    montar_prompt_aura_ncf,
    aura_reflector_analisar_interacao, get_live_memory_influence_pilar4
)

# Use TYPE_CHECKING to avoid circular import errors at runtime
if TYPE_CHECKING:
    from ceaf_adapter import CEAFAgentAdapter
    from ceaf_system.Integration import CEAFSystem

# Check for CEAF system availability
try:
    from ceaf_system.Integration import CEAFSystem

    CEAF_AVAILABLE = True
    print("✓ CEAF system found and loaded. Multi-tenant isolation is enabled.")
except ImportError as e: # <-- MODIFIED
    CEAF_AVAILABLE = False
    # --- THIS IS THE CRITICAL ADDITION ---
    print(f"❌ CEAF system failed to load. The specific error was: {e}")
    print("ℹ Continuing with NCF only. To enable CEAF, run: pip install sentence-transformers scikit-learn faiss-cpu langgraph")
    # --- END ADDITION ---


    # Define a placeholder class if CEAF is not available to prevent NameErrors
    class CEAFSystem:
        pass

# --- NEW SPECIALIST AGENT IDs ---
TENSORART_SPECIALIST_AGENT_ID = "6fd6c38b-cc35-4fc9-8b47-bef2e9cbcbf7"
SPEECH_SPECIALIST_AGENT_ID = "c1a0b3a9-5d2f-4e8a-b8e6-9f7d2c1e0a4b"
MCP_SPECIALIST_AGENT_ID = "b2d5e8c3-4a1f-4b9e-9c2d-0a8f7e6b1d0c"
A2A_SPECIALIST_AGENT_ID = "a3e8d9f4-5b2e-4c8d-8a1f-9b0c7d6e5a4b"
# --- END NEW IDs ---
logger = logging.getLogger(__name__)

MODEL_COSTS = {
    # --- Free / Beta Tier (Nominal cost to prevent abuse) ---
    "openrouter/deepseek/deepseek-chat-v3-0324:free": 1,
    "openrouter/deepseek/deepseek-r1-0528:free": 1,
    "openrouter/horizon-beta": 1,

    # --- Economy Tier (Very low API cost, good profit margin) ---
    "openrouter/mistralai/mistral-nemo": 2,
    "openrouter/openai/gpt-4o-mini": 2,

    # --- Standard Tier (Good balance, where most users should start) ---
    "openrouter/qwen/qwen3-30b-a3b-instruct-2507": 4,
    "openrouter/deepseek/deepseek-r1-0528": 4,
    "openrouter/google/gemini-2.0-flash-001": 4,
    "openrouter/meta-llama/llama-4-maverick": 5,
    "openrouter/openai/gpt-4.1-mini": 7,
    "openrouter/deepseek/deepseek-chat-v3-0324": 8,
    "openrouter/google/gemini-2.5-flash": 9,

    # --- Premium Tier (Significant cost jump, high profit margin) ---
    "openrouter/openai/gpt-4o": 25,
    "openrouter/openai/gpt-4.1": 25,
    "openrouter/google/gemini-2.5-pro": 25,

    # --- Elite Tier (Highest API cost, priced for paying customers) ---
    "openrouter/anthropic/claude-3.7-sonnet": 40,
    "openrouter/anthropic/claude-sonnet-4": 50,
    "openrouter/x-ai/grok-4": 50,

    # --- Default Cost ---
    "default": 15
}

MODEL_API_COSTS_USD = {
    # Free Models
    "deepseek/deepseek-chat-v3-0324:free": (0.0, 0.0),
    "deepseek/deepseek-r1-0528:free": (0.0, 0.0),
    "openrouter/horizon-beta": (0.0, 0.0),

    # Paid Models (Input Cost, Output Cost)
    "openrouter/mistralai/mistral-nemo": (0.008, 0.05),
    "openrouter/openai/gpt-4o-mini": (0.15, 0.60),
    "openrouter/qwen/qwen3-30b-a3b-instruct-2507": (0.20, 0.80),
    "openrouter/deepseek/deepseek-r1-0528": (0.272, 0.272),
    "openrouter/google/gemini-2.0-flash-001": (0.10, 0.40),
    "openrouter/meta-llama/llama-4-maverick": (0.15, 0.60),
    "openrouter/openai/gpt-4.1-mini": (0.40, 1.60),
    "openrouter/deepseek/deepseek-chat-v3-0324": (0.25, 0.85),
    "openrouter/google/gemini-2.5-flash": (0.30, 2.50),
    "openrouter/openai/gpt-4o": (1.5, 6.0),
    "openrouter/openai/gpt-4.1": (2.0, 8.0),
    "openrouter/google/gemini-2.5-pro": (1.25, 10.0),
    "openrouter/anthropic/claude-3.7-sonnet": (3.0, 15.0),
    "openrouter/anthropic/claude-sonnet-4": (3.0, 15.0),
    "openrouter/x-ai/grok-4": (3.0, 15.0),
    "default": (0.5, 1.5) # A safe default
}

# --- NEW: EXPANDED UserIntent ---
class UserIntent(str, Enum):
    CONVERSATION = "conversation"
    IMAGE_GENERATION = "image_generation"
    SPEECH_PROCESSING = "speech_processing"
    FILE_SEARCH = "file_search"
    MCP_TASK = "mcp_task"
    A2A_TASK = "a2a_task"

# --- NEW: EXPANDED INTENT_ROUTER_INSTRUCTION ---
INTENT_ROUTER_INSTRUCTION = f"""
You are an ultra-fast, efficient intent router. Your only job is to classify the user's request into one of the following categories.
Respond with ONLY the category name and nothing else.

Categories:
- `{UserIntent.CONVERSATION.value}`: For general chat, questions, discussions, storytelling, or any request that is not a specific command below.
- `{UserIntent.IMAGE_GENERATION.value}`: If the user explicitly asks to create, generate, draw, or make an image, picture, or photo.
- `{UserIntent.SPEECH_PROCESSING.value}`: If the user asks to generate audio, create a voice, read text aloud, or perform text-to-speech.
- `{UserIntent.FILE_SEARCH.value}`: If the user is asking a question that clearly requires looking up information in an uploaded document, PDF, or file.
- `{UserIntent.MCP_TASK.value}`: If the user mentions using an 'MCP server', 'MCP tool', or names a specific service known to use MCP (like GitHub, Sentry, Linear, Fetch).
- `{UserIntent.A2A_TASK.value}`: If the user asks to find, discover, contact, or interact with another 'agent', or mentions 'A2A' or 'AIRA Hub'.
"""


@dataclass
class AgentConfig:
    agent_id: str
    user_id: str
    name: str
    persona: str
    detailed_persona: str
    model: str = "openrouter/openai/gpt-4o-mini"
    memory_path: Optional[str] = None
    created_at: Optional[datetime] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    allow_live_memory_contribution: bool = False
    allow_live_memory_influence: bool = False


class AgentManager:
    """Manages multiple, isolated agent instances (NCF or CEAF) for different users."""

    def __init__(self, base_storage_path: str = None, db_repo: AgentRepository = None):
        """
        Initializes the AgentManager.
        FIXED: Automatically calculates the absolute path for agent_storage
        to be independent of the current working directory.
        """
        if base_storage_path is None:
            project_root = Path(__file__).parent.resolve()
            base_storage_path = project_root / "agent_storage"
            logger.info(f"No base_storage_path provided. Defaulting to absolute path: {base_storage_path}")

        self.base_storage_path = Path(base_storage_path)
        self.db_repo = db_repo
        self.base_storage_path.mkdir(exist_ok=True)

        self._active_agents: Dict[str, Any] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}

        self._ceaf_systems: Dict[str, 'CEAFSystem'] = {}

        self._load_agent_configs()


    def create_agent(self, user_id: str, name: str, persona: str,
                     detailed_persona: str, model: Optional[str] = None,
                     system_type: str = "ncf", settings: Optional[Dict[str, Any]] = None) -> str:
        """Creates a new agent (NCF, CEAF, or Blank). Delegates to helper methods."""
        if system_type == "ceaf":
            if not CEAF_AVAILABLE:
                raise ValueError("CEAF system is not available. Cannot create a CEAF agent.")
            return self._create_ceaf_agent(user_id, name, persona, detailed_persona, model)
        else:  # Default to NCF or Blank Tiers
            return self._create_ncf_agent(user_id, name, persona, detailed_persona, model, settings)

    def _create_ncf_agent(self, user_id: str, name: str, persona: str,
                          detailed_persona: str, model: Optional[str] = None,
                          settings: Optional[Dict[str, Any]] = None) -> str:
        """Cria a configuração para um agente NCF ou um agente 'Branco' mais simples."""
        agent_id = str(uuid.uuid4())
        agent_path = self.base_storage_path / user_id / agent_id
        agent_path.mkdir(parents=True, exist_ok=True)

        # O settings agora controla o "nível" do agente
        agent_settings = settings or {}
        agent_settings.setdefault("system_type", "ncf")  # Garante que o tipo exista

        # Lógica para agentes "Brancos"
        if agent_settings.get("tier") == "blank_memory":
            # Agente com memória, mas sem NCF. O prompt será o detailed_persona.
            agent_settings["use_ncf_prompting"] = False
            memory_path = str(agent_path / "memory_blossom.json")
        elif agent_settings.get("tier") == "blank_rag":
            # Agente sem memória, só RAG.
            agent_settings["use_ncf_prompting"] = False
            agent_settings["use_memory_blossom"] = False
            memory_path = None  # Sem memória
        else:
            # Agente NCF padrão
            agent_settings["use_ncf_prompting"] = True
            memory_path = str(agent_path / "memory_blossom.json")

        config = AgentConfig(
            agent_id=agent_id,
            user_id=user_id,
            name=name,
            persona=persona,
            detailed_persona=detailed_persona,  # Para agentes "Brancos", este é o system prompt direto
            model=model or "openrouter/openai/gpt-4o-mini",
            memory_path=memory_path,
            created_at=datetime.now(),
            settings=agent_settings
        )

        self._save_agent_config(config)
        self._agent_configs[agent_id] = config
        logger.info(f"Created agent '{name}' (Tier: {agent_settings.get('tier', 'ncf')}) for user {user_id}.")
        return agent_id

    def _create_ceaf_agent(self, user_id: str, name: str, persona: str,
                           detailed_persona: str, model: Optional[str] = None) -> str:
        """Creates the configuration for an ISOLATED CEAF agent."""
        agent_id = str(uuid.uuid4())
        agent_path = self.base_storage_path / user_id / agent_id
        agent_path.mkdir(parents=True, exist_ok=True)

        persistence_path = str(agent_path)
        ceaf_config = {
            "persistence_path": persistence_path,
            "memory": {"embedding_model": "all-MiniLM-L6-v2", "max_clusters": 1000},
            "mcl": {"history_window": 100},
            "orchestrator": {"default_model": model or "openrouter/openai/gpt-4o-mini"},
            "auto_save_interval": 5,
            "aura_analysis_interval": 10
        }

        config = AgentConfig(
            agent_id=agent_id,
            user_id=user_id,
            name=name,
            persona=persona,
            detailed_persona=detailed_persona,
            model=model or "openrouter/openai/gpt-4o-mini",
            memory_path=str(Path(persistence_path) / "memory_state.json"),
            created_at=datetime.now(),
            settings={
                "system_type": "ceaf",
                "ceaf_config": ceaf_config,
                "capabilities": [
                    "adaptive_memory_architecture", "metacognitive_control_loop",
                    "narrative_coherence_identity", "universal_reflective_analyzer"
                ]
            }
        )

        self._save_agent_config(config)
        self._agent_configs[agent_id] = config
        logger.info(f"Created CEAF agent '{name}' (ID: {agent_id}). Instance will be created on first use.")
        return agent_id

    def get_agent_instance(self, agent_id: str) -> Optional[Any]:
        if agent_id not in self._agent_configs:
            logger.warning(f"Agent {agent_id} not in memory cache. Forcing a reload from filesystem.")
            self._load_agent_configs()
            if agent_id not in self._agent_configs:
                logger.error(f"Agent {agent_id} still not found after reload. The config file may be missing or corrupted.")
                return None

        config = self._agent_configs[agent_id]

        if config.settings.get("system_type") == "ceaf":
            if not CEAF_AVAILABLE:

                from fastapi import HTTPException
                logger.error(f"CEAF requested for agent {agent_id}, but the system is not available on this server.")
                raise HTTPException(
                    status_code=503,
                    detail="This agent requires the CEAF system, which is not available or failed to load on the server."
                )

            return self._get_ceaf_instance_with_adapter(agent_id, config)
        else:
            return self._get_ncf_instance(agent_id, config)

    def _get_ceaf_instance_with_adapter(self, agent_id: str, config: AgentConfig) -> Optional[Any]:
        if agent_id in self._active_agents:
            return self._active_agents[agent_id]

        ceaf_system = self._get_or_create_ceaf_system(agent_id, config)
        if ceaf_system is None:
            logger.error(f"Failed to create isolated CEAF system for agent {agent_id}")
            return None

        from ceaf_adapter import CEAFAgentAdapter
        adapter = CEAFAgentAdapter(ceaf_system, config, self.db_repo)
        self._active_agents[agent_id] = adapter
        logger.info(f"✓ Activated isolated CEAF agent via adapter: {config.name} (ID: {agent_id})")
        return adapter

    def get_agent_system_type(self, agent_id: str) -> str:
        config = self._agent_configs.get(agent_id)
        if config:
            return config.settings.get("system_type", "ncf")
        return "ncf"

    def _get_or_create_ceaf_system(self, agent_id: str, config: AgentConfig) -> Optional['CEAFSystem']:
        """Gets a cached CEAF system or creates a new, isolated one."""
        # Check cache first.
        if agent_id in self._ceaf_systems:
            return self._ceaf_systems[agent_id]

        ceaf_config = config.settings.get("ceaf_config", {})

        # Ensure the persistence path is correctly set for isolation.
        if "persistence_path" not in ceaf_config:
            agent_path = self.base_storage_path / config.user_id / agent_id
            ceaf_config["persistence_path"] = str(agent_path)
            logger.warning(f"Persistence path missing for CEAF agent {agent_id}. Defaulting to: {agent_path}")

        # Create a NEW, ISOLATED CEAF system instance.
        ceaf_system = CEAFSystem(ceaf_config)

        # Cache the isolated system using its agent_id as the key.
        self._ceaf_systems[agent_id] = ceaf_system
        logger.info(f"✓ Created new ISOLATED CEAF system for agent {agent_id}")
        return ceaf_system

    def _get_ncf_instance(self, agent_id: str, config: AgentConfig) -> 'NCFAuraAgentInstance':
        if agent_id in self._active_agents:
            return self._active_agents[agent_id]

        use_memory = config.settings.get("use_memory_blossom", True)
        memory_path_for_init = config.memory_path if use_memory else None

        instance = NCFAuraAgentInstance(config, self.db_repo, self, memory_path_for_init)
        self._active_agents[agent_id] = instance
        return instance

    def list_user_agents(self, user_id: str) -> List[AgentConfig]:
        """Lists all agents for a specific user."""
        return [config for config in self._agent_configs.values() if config.user_id == user_id]

    def delete_agent(self, agent_id: str, user_id: str) -> bool:
        """Deletes an agent and cleans up all its resources (NCF or CEAF)."""
        if agent_id not in self._agent_configs or self._agent_configs[agent_id].user_id != user_id:
            return False

        config = self._agent_configs[agent_id]


        if agent_id in self._ceaf_systems:
            try:
                ceaf_instance = self._ceaf_systems[agent_id]
                if hasattr(ceaf_instance, 'shutdown'):
                    ceaf_instance.shutdown()
                del self._ceaf_systems[agent_id]
                logger.info(f"✓ Cleaned up and shut down cached CEAF system for agent {agent_id}")
            except Exception as e:
                logger.warning(f"Error during CEAF system cleanup: {e}")

        if agent_id in self._active_agents:
            del self._active_agents[agent_id]

        # Delete agent's dedicated storage directory.
        agent_path = self.base_storage_path / config.user_id / agent_id
        if agent_path.exists():
            shutil.rmtree(agent_path)

        # Delete agent's configuration file.
        del self._agent_configs[agent_id]
        config_path = self.base_storage_path / config.user_id / f"{agent_id}.json"
        if config_path.exists():
            config_path.unlink()

        logger.info(f"Deleted agent {agent_id} and all its resources for user {user_id}.")
        return True

    def _save_agent_config(self, config: AgentConfig):
        """Saves an agent's configuration to a JSON file."""
        user_path = self.base_storage_path / config.user_id
        user_path.mkdir(exist_ok=True)
        config_path = user_path / f"{config.agent_id}.json"

        # Create a serializable dictionary from the dataclass
        config_dict = {
            'agent_id': config.agent_id,
            'user_id': config.user_id,
            'name': config.name,
            'persona': config.persona,
            'detailed_persona': config.detailed_persona,
            'model': config.model,
            'memory_path': config.memory_path,
            'created_at': config.created_at.isoformat() if config.created_at else None,
            'settings': config.settings or {}
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def _load_agent_configs(self):
        """Loads all agent configurations from the file system on startup."""
        for user_dir in self.base_storage_path.iterdir():
            if not user_dir.is_dir():
                continue
            for config_file in user_dir.glob("*.json"):
                if "memory" in config_file.name:  # Skip memory files
                    continue
                try:
                    with open(config_file, 'r') as f:
                        data = json.load(f)

                    # Basic validation
                    required_keys = ['agent_id', 'user_id', 'name', 'persona', 'detailed_persona', 'created_at']
                    if not all(k in data for k in required_keys):
                        logger.warning(f"Skipping config file {config_file}: missing essential keys.")
                        continue

                    # ==================== START: ROBUST MODEL NAME FIX ====================
                    # Get the model name from the file, falling back to the corrected default
                    model = data.get('model', 'openrouter/openai/gpt-4o-mini')

                    # Automatically prefix legacy model names to ensure they use OpenRouter
                    # This makes the system resilient to old agent configurations.
                    if "openrouter/" not in model and ":free" not in model:
                        logger.warning(f"Found legacy model name '{model}' in config {config_file.name}. Auto-prefixing with 'openrouter/'.")
                        model = f"openrouter/{model}"
                    # ===================== END: ROBUST MODEL NAME FIX =====================


                    config = AgentConfig(
                        agent_id=data['agent_id'],
                        user_id=data['user_id'],
                        name=data['name'],
                        persona=data['persona'],
                        detailed_persona=data['detailed_persona'],
                        model=model, # Use the potentially corrected model name
                        memory_path=data.get('memory_path'),
                        created_at=datetime.fromisoformat(data['created_at']),
                        settings=data.get('settings', {})
                    )
                    self._agent_configs[config.agent_id] = config
                except Exception as e:
                    logger.error(f"Error loading config file {config_file}: {e}")

    def get_agent_stats(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive stats for an agent (works for both NCF and CEAF)"""
        try:
            agent = self.get_agent_instance(agent_id)
            if not agent:
                return None

            if hasattr(agent, 'get_enhanced_stats'):
                return agent.get_enhanced_stats()
            else:
                # Basic stats for agents without enhanced stats
                config = self._agent_configs.get(agent_id)
                return {
                    "agent_id": agent_id,
                    "agent_name": config.name if config else "Unknown",
                    "system_type": config.settings.get("system_type", "unknown") if config else "unknown",
                    "created_at": config.created_at.isoformat() if config and config.created_at else None
                }

        except Exception as e:
            logger.error(f"Error getting agent stats: {e}")
            return None

    def is_ceaf_available(self) -> bool:
        """Check if CEAF system is available"""
        return CEAF_AVAILABLE

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the agent management system"""
        return {
            "total_agents": len(self._agent_configs),
            "active_agents": len(self._active_agents),
            "ceaf_systems": len(self._ceaf_systems),
            "ceaf_available": CEAF_AVAILABLE,
            "system_types": {
                "ncf": len([c for c in self._agent_configs.values() if c.settings.get("system_type") == "ncf"]),
                "ceaf": len([c for c in self._agent_configs.values() if c.settings.get("system_type") == "ceaf"])
            }
        }

    @property
    def agent_configs(self):
        return self._agent_configs


class NCFAuraAgentInstance:
    """
    Instância de agente que pode ser um Orquestrador NCF ou um Especialista em Tensor.Art.
    O comportamento é determinado pelo seu agent_id.
    CORRIGIDO: A ferramenta de delegação agora usa ToolContext para obter o session_id,
    removendo a ambiguidade para o LLM. O nome da ferramenta também foi simplificado.
    """

    def __init__(self, config: AgentConfig, db_repo: AgentRepository, agent_manager: 'AgentManager',
                 memory_path: Optional[str] = None):
        self.config = config
        self.db_repo = db_repo
        self.agent_manager = agent_manager
        self.model = LiteLlm(model=config.model)
        self.app_name = f"NCFAura_{config.agent_id}"

        # <<< CORREÇÃO AQUI: Lógica para agentes sem memória
        use_memory = self.config.settings.get("use_memory_blossom", True)

        if use_memory:
            try:
                from enhanced_memory_system import EnhancedMemoryBlossom
                self.memory_blossom = EnhancedMemoryBlossom(
                    persistence_path=memory_path or config.memory_path,  # Usa o path passado ou o do config
                    enable_adaptive_rag=True
                )
            except ImportError:
                self.memory_blossom = MemoryBlossom(persistence_path=memory_path or config.memory_path)

            self.memory_connector = MemoryConnector(self.memory_blossom)
            self.memory_blossom.set_memory_connector(self.memory_connector)
        else:
            self.memory_blossom = None
            self.memory_connector = None

        # --- DYNAMIC TOOL INITIALIZATION BASED ON ROLES ---
        self.tools = []
        roles = self.config.settings.get("roles", ["orchestrator"])

        if "orchestrator" in roles:
            print(f"Initializing agent '{config.name}' with Orchestrator tools.")

            if use_memory:
                self.tools.extend([
                    FunctionTool(func=self._create_add_memory_func()),
                    FunctionTool(func=self._create_recall_memories_func()),
                ])


                #  The router handles this now.
                self.tools.extend([
                    FunctionTool(func=self._create_search_files_func()),

                ])


        if "specialist" in roles:
            if "image_generation" in roles:
                print(f"Initializing agent '{config.name}' with Image Specialist tools.")
                self.tensorart_client = TensorArtClient()
                self.tools.extend([
                    FunctionTool(func=self._create_generate_image_func()),
                    FunctionTool(func=self._create_check_job_status_func()),
                    FunctionTool(func=self._create_calculate_image_cost_func())
                ])
            if "speech_processing" in roles:
                print(f"Initializing agent '{config.name}' with Speech Specialist tools.")
                self.speech_client = SpeechClient()
                self.tools.extend([
                    FunctionTool(func=self.speech_client.generate_speech),
                    FunctionTool(func=self.speech_client.list_available_models)
                ])
            # --- NEW: MCP Specialist Role ---
            if "mcp_specialist" in roles:
                print(f"Initializing agent '{config.name}' with MCP Specialist tools.")
                self.mcp_client = MCPClient()
                self.tools.extend([
                    FunctionTool(func=self.mcp_client.mcp_connect),
                    FunctionTool(func=self.mcp_client.mcp_call_tool),
                    FunctionTool(func=self.mcp_client.mcp_disconnect)
                ])
            # --- NEW: A2A Specialist Role ---
            if "a2a_specialist" in roles:
                print(f"Initializing agent '{config.name}' with A2A Specialist tools.")
                # Make the Hub URL configurable via environment variable or default
                aira_hub_url = os.getenv("AIRA_HUB_URL", "https://airahub2.onrender.com")
                self.a2a_client = A2AClient(hub_url=aira_hub_url)
                self.tools.extend([
                    FunctionTool(func=self.a2a_client.discover_agents),
                    FunctionTool(func=self.a2a_client.send_task),
                    FunctionTool(func=self.a2a_client.get_task_status)
                ])

        self.adk_agent = self._create_ncf_adk_agent()
        self.session_service = InMemorySessionService()
        self.runner = Runner(agent=self.adk_agent, app_name=self.app_name, session_service=self.session_service)
        self.active_sessions: Dict[str, str] = {}

    def _create_ncf_adk_agent(self) -> LlmAgent:
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.config.name)
        if not re.match(r'^[a-zA-Z_]', sanitized_name):
            sanitized_name = '_' + sanitized_name

        # Determine instruction and tools based on roles
        roles = self.config.settings.get("roles", ["orchestrator"])
        instruction = NCF_AGENT_INSTRUCTION  # Default for orchestrator

        # --- NEW: Select instruction based on specialist role ---
        if "image_generation" in roles:
            instruction = TENSORART_SPECIALIST_INSTRUCTION
        elif "speech_processing" in roles:
            instruction = SPEECH_SPECIALIST_INSTRUCTION
        elif "mcp_specialist" in roles:
            instruction = MCP_SPECIALIST_INSTRUCTION
        elif "a2a_specialist" in roles:
            instruction = A2A_SPECIALIST_INSTRUCTION
        # --- END NEW ---
        
        return LlmAgent(
            name=sanitized_name,
            model=self.model,
            instruction=instruction,
            tools=self.tools  # Use the dynamically built list of tools
        )

    def _create_delegate_to_specialist_func(self, specialist_id: str, task_type: str):
        async def delegate_task(task_description: str, tool_context: Optional[Any] = None) -> Dict[str, Any]:
            f"""
               Use this tool for any tasks related to {task_type}.
               Provide a clear and complete description of the user's request.
               For follow-up interactions, your input will be combined with the conversation history.
               """
            try:
                session_id = None
                full_context_for_specialist = task_description  # Start with the current instruction

                # <<< CORREÇÃO AQUI: Lógica de Concatenação de Contexto >>>
                if tool_context and hasattr(tool_context, 'invocation_context') and hasattr(
                        tool_context.invocation_context, 'session'):
                    session_id = tool_context.invocation_context.session.id
                    session_history = tool_context.invocation_context.session.history

                    # Reconstruct a concise history of this specific task for the specialist
                    task_related_history = []
                    # The history is in ADK Content format (role, parts)
                    for adk_content in reversed(session_history):
                        # Stop when we see a successful specialist response, indicating a new sub-task started
                        if adk_content.role == "tool" and "specialist_response" in adk_content.parts[0].text:
                            break
                        # Find previous user prompts related to this tool
                        if adk_content.role == "user":
                            # The user's prompt is inside a larger NCF prompt. We extract the relevant part.
                            match = re.search(r'Usuário: "([^"]+)"', adk_content.parts[0].text)
                            if match:
                                task_related_history.append(f"User previously said: '{match.group(1)}'")

                    if task_related_history:
                        # Reverse to get chronological order and join with current task
                        task_related_history.reverse()
                        history_str = "\n".join(task_related_history)
                        full_context_for_specialist = f"Here is the conversation history for this task:\n{history_str}\n\nLatest user instruction: '{task_description}'"
                        logger.info(f"Passing full context to specialist: {full_context_for_specialist}")

                if not session_id:
                    logger.warning("Could not retrieve session_id from ToolContext. Using a fallback.")
                    session_id = f"fallback_{uuid.uuid4()}"

                specialist_agent = self.agent_manager.get_agent_instance(specialist_id)
                if not specialist_agent:
                    return {"status": "error", "message": f"The {task_type} specialist agent is not active."}

                # Pass the full, reconstructed context to the specialist
                response_dict = await specialist_agent.process_message(
                    user_id=self.config.user_id,
                    session_id=f"specialist_subsession_{session_id}",
                    message=full_context_for_specialist
                )
                return {"status": "success", "specialist_response": response_dict.get("response")}
            except Exception as e:
                logger.error(f"Error in {task_type} delegation tool: {e}", exc_info=True)
                return {"status": "error", "message": f"Delegation to {task_type} specialist failed: {str(e)}"}

        delegate_task.__name__ = f"delegate_{task_type}_task"
        return delegate_task

    # --- Ferramentas Diretas da API (Apenas para o Especialista) ---
    def _build_tensorart_stages(self, prompt: str, negative_prompt: Optional[str], sd_model: str,
                                sampler: str, steps: int, cfg_scale: float, width: int, height: int) -> List[
        Dict[str, Any]]:
        """Helper para construir a lista de 'stages' para a API Tensor.Art."""
        stages = [
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": {"seed": -1, "count": 1}
            },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "prompts": [{"text": prompt}],
                    "negativePrompts": [{"text": negative_prompt or ""}],
                    "sd_model": sd_model,
                    "sampler": sampler,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "width": width,
                    "height": height,
                    "clip_skip": 2
                }
            }
        ]
        return stages

    def _create_calculate_image_cost_func(self):
        # This tool is now more powerful and flexible.
        def calculate_cost_from_stages(stages_json: str) -> Dict[str, Any]:
            """
            Calculates the estimated credit cost for an image generation job by providing a JSON string of the full 'stages' workflow.
            The stages_json argument must be a string containing a valid JSON list of stage objects.
            """
            try:
                stages = json.loads(stages_json)
                if not isinstance(stages, list):
                    return {"status": "error",
                            "message": "The 'stages_json' argument must be a string representing a JSON list."}

                cost = self.tensorart_client.calculate_credits(stages)
                return {"status": "success", "estimated_cost_credits": cost}
            except json.JSONDecodeError:
                return {"status": "error", "message": "Invalid JSON format in stages_json string."}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        return calculate_cost_from_stages

    def _create_generate_image_func(self):
        # This tool now directly accepts the stage workflow from the LLM.
        def generate_image_from_stages(stages_json: str) -> Dict[str, Any]:
            """
            Submits an image generation job to the Tensor.Art API by providing a JSON string of the full 'stages' workflow.
            Use this ONLY after the user has confirmed the cost.
            """
            try:
                stages = json.loads(stages_json)
                if not isinstance(stages, list):
                    return {"status": "error",
                            "message": "The 'stages_json' argument must be a string representing a JSON list."}

                result = self.tensorart_client.submit_job(stages)
                return result
            except json.JSONDecodeError:
                return {"status": "error", "message": "Invalid JSON format in stages_json string."}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        return generate_image_from_stages

    def _create_check_job_status_func(self):
        def check_image_job_status(job_id: str) -> Dict[str, Any]:
            """Checks the status of a previously submitted image generation job."""
            try:
                status = self.tensorart_client.check_job_status(job_id)
                return status
            except Exception as e:
                return {"status": "error", "message": str(e)}

        return check_image_job_status


    def _create_search_files_func(self):
        """Creates the search_agent_files function for the agent's tools."""

        def search_agent_files(query: str, top_k: int = 3) -> Dict[str, Any]:
            """
            Searches through the user-uploaded files for this specific agent.
            Use this tool to answer questions based on documents, PDFs, or text files
            the user has provided to you.
            """
            try:
                # O agent_storage_path é construído dinamicamente
                agent_storage_path = Path("agent_storage") / self.config.user_id / self.config.agent_id

                results = search_in_agent_files(agent_storage_path, query, top_k)

                return {
                    "status": "success",
                    "results_found": len(results),
                    "search_results": results
                }
            except Exception as e:
                logger.error(f"Error searching agent files for {self.config.agent_id}: {e}")
                return {"status": "error", "message": str(e)}

        return search_agent_files

    def _create_add_memory_func(self):
        """Create the add_memory function for the agent's tools"""

        def add_memory(content: str, memory_type: str,
                       emotion_score: float = 0.0,
                       initial_salience: float = 0.5,
                       metadata_json: Optional[str] = None,
                       domain_context: str = "general",
                       performance_score: float = 0.5,
                       tool_context=None) -> Dict[str, Any]:
            try:
                custom_metadata = json.loads(metadata_json) if metadata_json else {}
                custom_metadata['agent_id'] = self.config.agent_id
                custom_metadata['agent_name'] = self.config.name
                custom_metadata['domain_context'] = domain_context
                custom_metadata['performance_score'] = performance_score

                # Check if enhanced memory system is available
                if hasattr(self.memory_blossom, 'enable_adaptive_rag') and self.memory_blossom.enable_adaptive_rag:
                    memory = self.memory_blossom.add_memory(
                        content=content,
                        memory_type=memory_type,
                        custom_metadata=custom_metadata,
                        emotion_score=emotion_score,
                        initial_salience=initial_salience,
                        performance_score=performance_score,
                        domain_context=domain_context
                    )
                    message_suffix = " (Enhanced with Adaptive RAG)"
                else:
                    memory = self.memory_blossom.add_memory(
                        content=content,
                        memory_type=memory_type,
                        custom_metadata=custom_metadata,
                        emotion_score=emotion_score,
                        initial_salience=initial_salience
                    )
                    message_suffix = " (Standard MemoryBlossom)"

                self.memory_blossom.save_memories()

                return {
                    "status": "success",
                    "memory_id": memory.id,
                    "message": f"Memory stored successfully for {self.config.name} (ID: {memory.id}){message_suffix}",
                    "adaptive_rag_enabled": hasattr(self.memory_blossom, 'enable_adaptive_rag')
                }
            except Exception as e:
                logger.error(f"Error adding memory for agent {self.config.agent_id}: {e}")
                return {"status": "error", "message": str(e)}

        return add_memory

    def _create_recall_memories_func(self):
        """Create the recall_memories function for the agent's tools"""

        def recall_memories(query: str,
                            target_memory_types_json: Optional[str] = None,
                            top_k: int = 3,
                            domain_context: str = "general",
                            tool_context=None) -> Dict[str, Any]:
            try:
                target_types = None
                if target_memory_types_json:
                    target_types = json.loads(target_memory_types_json)

                conversation_history_for_retrieval = None
                if tool_context and tool_context.state and 'conversation_history' in tool_context.state:
                    conversation_history_for_retrieval = tool_context.state['conversation_history']

                # Use enhanced retrieval if available
                if hasattr(self.memory_blossom, 'adaptive_retrieve_memories'):
                    memories = self.memory_blossom.adaptive_retrieve_memories(
                        query=query,
                        target_memory_types=target_types,
                        domain_context=domain_context,
                        top_k=top_k,
                        use_performance_weighting=True,
                        conversation_context=conversation_history_for_retrieval
                    )
                    retrieval_method = "Enhanced Adaptive RAG"
                else:
                    memories = self.memory_blossom.retrieve_memories(
                        query=query,
                        target_memory_types=target_types,
                        top_k=top_k,
                        conversation_context=conversation_history_for_retrieval
                    )
                    retrieval_method = "Standard MemoryBlossom"

                return {
                    "status": "success",
                    "count": len(memories),
                    "memories": [mem.to_dict() for mem in memories],
                    "retrieval_method": retrieval_method,
                    "adaptive_rag_enabled": hasattr(self.memory_blossom, 'adaptive_retrieve_memories')
                }
            except Exception as e:
                logger.error(f"Error recalling memories for agent {self.config.agent_id}: {e}")
                return {"status": "error", "message": str(e)}

        return recall_memories

    async def process_message(self, user_id: str, session_id: str, message: str,
                              session_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        # ======================= UNIVERSAL SETUP: DYNAMIC COST LOGIC =======================
        CREDITS_PER_DOLLAR = 500
        PROFIT_MARKUP = 3.0

        def calculate_credit_cost(model_name: str, input_tokens: int, output_tokens: int) -> int:
            prices = MODEL_API_COSTS_USD.get(model_name, MODEL_API_COSTS_USD["default"])
            input_price_per_m, output_price_per_m = prices
            cost_usd = ((input_tokens / 1_000_000) * input_price_per_m) + \
                       ((output_tokens / 1_000_000) * output_price_per_m)
            final_cost_usd = cost_usd * PROFIT_MARKUP
            credit_cost = final_cost_usd * CREDITS_PER_DOLLAR
            return max(1, int(credit_cost))

        model_to_use = self.config.model
        if session_overrides and 'model' in session_overrides:
            model_to_use = session_overrides['model']
            logger.info(f"Using session override model for agent '{self.config.name}': {model_to_use}")

        # --- UNIVERSAL SETUP: CREDIT CHECK GATEKEEPER ---
        with self.db_repo.SessionLocal() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return {"response": "Authentication error. Could not verify your user account.", "prompt": "",
                        "domain_context": "error"}
            if user.credits < 1:
                return {"response": "You are out of credits! Please add more to continue.", "prompt": "",
                        "domain_context": "billing"}

        # ======================================================================
        # ==================== NEW: INTENT ROUTING LAYER =======================
        # ======================================================================
        logger.info(f"Routing intent for message: '{message[:50]}...'")
        try:
            # Use a fast, cheap model for routing.
            router_model = "openrouter/openai/gpt-4o-mini"
            response = await litellm.acompletion(
                model=router_model,
                messages=[
                    {"role": "system", "content": INTENT_ROUTER_INSTRUCTION},
                    {"role": "user", "content": message}
                ],
                temperature=0.0,
                max_tokens=15
            )
            intent_str = response.choices[0].message.content.strip().lower()
            intent = UserIntent(intent_str)
            logger.info(f"Intent classified as: {intent.value}")
        except (ValueError, KeyError, IndexError) as e:
            logger.warning(f"Intent classification failed ('{e}'). Defaulting to conversation.")
            intent = UserIntent.CONVERSATION

        # --- PATH A: SPECIALIST TASKS (Image/Speech/MCP/A2A etc.) ---
        if intent in [UserIntent.IMAGE_GENERATION, UserIntent.SPEECH_PROCESSING, UserIntent.MCP_TASK, UserIntent.A2A_TASK]:
            logger.info(f"Handling '{intent.value}' directly via specialist delegation.")
            
            specialist_map = {
                UserIntent.IMAGE_GENERATION: (TENSORART_SPECIALIST_AGENT_ID, "image"),
                UserIntent.SPEECH_PROCESSING: (SPEECH_SPECIALIST_AGENT_ID, "speech"),
                UserIntent.MCP_TASK: (MCP_SPECIALIST_AGENT_ID, "mcp"),
                UserIntent.A2A_TASK: (A2A_SPECIALIST_AGENT_ID, "a2a"),
            }
            specialist_id, task_type = specialist_map[intent]

            specialist_agent = self.agent_manager.get_agent_instance(specialist_id)
            if not specialist_agent:
                return {"response": f"The {task_type} specialist is unavailable.", "prompt": "Specialist Error",
                        "domain_context": "error"}

            # Directly invoke the specialist
            specialist_session_id = f"specialist_subsession_{session_id}"
            response_dict = await specialist_agent.process_message(
                user_id=self.config.user_id,
                session_id=specialist_session_id,
                message=message
            )
            # Cost is handled by the specialist's own process_message call.
            return {
                "response": response_dict.get("response"),
                "prompt": f"Routed to {task_type} specialist.",
                "domain_context": intent.value
            }

        # --- PATH B: CONVERSATIONAL TASK (The original NCF workflow) ---
        elif intent in [UserIntent.CONVERSATION, UserIntent.FILE_SEARCH]:
            # The existing, powerful NCF workflow runs only when needed.
            logger.info("Proceeding with full NCF conversational workflow.")
            original_model = self.adk_agent.model
            try:
                total_input_tokens = 0
                total_output_tokens = 0
                self.adk_agent.model = LiteLlm(model=model_to_use)

                ncf_session_key = f"{user_id}_{session_id}"
                if ncf_session_key not in self.active_sessions:
                    self.active_sessions[ncf_session_key] = {'conversation_history': [],
                                                             'foundation_narrative_turn_count': 0}
                current_session_state = self.active_sessions[ncf_session_key]

                adk_turn_session_id = f"turn_adk_{uuid.uuid4()}"
                await self.session_service.create_session(app_name=self.app_name, user_id=user_id,
                                                          session_id=adk_turn_session_id, state={})
                current_session_state['conversation_history'].append({'role': 'user', 'content': message})

                domain_context = await self._detect_domain_context(message, current_session_state)
                narrativa_fundamento = await get_narrativa_de_fundamento_pilar1(session_state=current_session_state,
                                                                                memory_blossom=self.memory_blossom,
                                                                                user_id=user_id,
                                                                                llm_instance=self.adk_agent.model,
                                                                                agent_name=self.config.name,
                                                                                agent_persona=self.config.persona)

                if hasattr(self.memory_blossom, 'adaptive_retrieve_memories'):
                    rag_memories = self.memory_blossom.adaptive_retrieve_memories(query=message,
                                                                                  domain_context=domain_context,
                                                                                  top_k=3,
                                                                                  use_performance_weighting=True,
                                                                                  conversation_context=current_session_state.get(
                                                                                      'conversation_history', []))
                else:
                    rag_memories = self.memory_blossom.retrieve_memories(query=message, top_k=3,
                                                                         conversation_context=current_session_state.get(
                                                                             'conversation_history', []))
                rag_info_list = [mem.to_dict() for mem in rag_memories]

                chat_history_str = format_chat_history_pilar3(
                    chat_history_list=current_session_state['conversation_history'])
                live_memory_influence = []
                if self.config.allow_live_memory_influence:
                    live_memory_influence = await get_live_memory_influence_pilar4(message)

                final_ncf_prompt = montar_prompt_aura_ncf(agent_name=self.config.name,
                                                          agent_detailed_persona=self.config.detailed_persona,
                                                          narrativa_fundamento=narrativa_fundamento,
                                                          informacoes_rag_list=rag_info_list,
                                                          chat_history_recente_str=chat_history_str,
                                                          live_memory_influence_list=live_memory_influence,
                                                          user_reply=message, domain_context=domain_context)

                adk_message = ADKContent(role="user", parts=[ADKPart(text=final_ncf_prompt)])
                response_text = ""
                run_usage_metadata = None

                async for event in self.runner.run_async(user_id=user_id, session_id=adk_turn_session_id,
                                                         new_message=adk_message):
                    if event.is_final_response():
                        if event.content and event.content.parts:
                            response_text = event.content.parts[0].text
                        if hasattr(event, 'metadata') and 'usage' in event.metadata:
                            run_usage_metadata = event.metadata['usage']
                        break

                response_text = response_text or f"({self.config.name} did not provide a response for this turn)"

                if run_usage_metadata:
                    total_input_tokens += run_usage_metadata.get('prompt_tokens', 0)
                    total_output_tokens += run_usage_metadata.get('completion_tokens', 0)
                else:
                    total_input_tokens += len(final_ncf_prompt) // 4
                    total_output_tokens += len(response_text) // 4

                current_session_state['conversation_history'].append({'role': 'assistant', 'content': response_text})
                self.active_sessions[ncf_session_key] = current_session_state

                final_cost = calculate_credit_cost(model_to_use, total_input_tokens, total_output_tokens)
                with self.db_repo.SessionLocal() as session:
                    user_to_update = session.query(User).filter(User.id == user_id).first()
                    if user_to_update:
                        if user_to_update.credits < final_cost:
                            return {"response": "You do not have enough credits for that response.",
                                    "prompt": final_ncf_prompt, "domain_context": "billing"}
                        user_to_update.credits -= final_cost
                        transaction = CreditTransaction(user_id=user_id, agent_id=self.config.agent_id,
                                                        amount=-final_cost, model_used=model_to_use,
                                                        description=f"Chat with {self.config.name} ({total_input_tokens} in, {total_output_tokens} out)")
                        session.add(transaction)
                        session.commit()

                return {"response": response_text, "prompt": final_ncf_prompt, "domain_context": domain_context}

            except Exception as e:
                logger.error(f"Error processing NCF message for {self.config.name}: {e}", exc_info=True)
                return {"response": f"Sorry, there was an error. As {self.config.name}, I'll do my best to help.",
                        "prompt": "Error", "domain_context": "error"}
            finally:
                self.adk_agent.model = original_model

        else:
            # Fallback for any unhandled intents
            return {"response": "I'm not sure how to handle that request type.", "prompt": "Intent Error",
                    "domain_context": "error"}

    async def _detect_domain_context(self, message: str, session_state: Dict[str, Any]) -> str:
        """Detect conversation domain from user message"""
        try:
            # Simple keyword-based detection (can be enhanced with ML)
            message_lower = message.lower()

            # Enhanced keyword detection with more comprehensive lists
            domain_keywords = {
                'physics': ['physics', 'quantum', 'mechanics', 'energy', 'force', 'wave', 'particle',
                            'momentum', 'velocity', 'acceleration', 'relativity', 'thermodynamics'],
                'mathematics': ['calculate', 'equation', 'solve', 'mathematics', 'algebra', 'geometry',
                                'calculus', 'derivative', 'integral', 'function', 'graph', 'variable'],
                'emotional_support': ['feel', 'sad', 'happy', 'worried', 'excited', 'frustrated', 'love',
                                      'anxious', 'depressed', 'angry', 'afraid', 'emotional', 'mood'],
                'nsfw_role_play': ['nsfw', 'sexy', 'erotic', 'desire', 'pleasure', 'te excita', 'fetiche', 'fetish',
                                   'role-play'],
                'creative_writing': ['story', 'write', 'creative', 'imagine', 'design', 'art', 'poem',
                                     'character', 'plot', 'narrative', 'fiction', 'novel'],
                'programming': ['code', 'programming', 'function', 'variable', 'algorithm', 'debug',
                                'software', 'python', 'javascript', 'html', 'css'],
                'science': ['biology', 'chemistry', 'experiment', 'hypothesis', 'research', 'study',
                            'scientific', 'theory', 'observation', 'data'],
                'health': ['health', 'medical', 'doctor', 'medicine', 'symptoms', 'treatment',
                           'illness', 'wellness', 'exercise', 'nutrition'],
                'education': ['learn', 'study', 'school', 'university', 'education', 'teaching',
                              'homework', 'assignment', 'exam', 'knowledge']
            }

            # Check for domain matches
            for domain, keywords in domain_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    logger.debug(
                        f"Domain '{domain}' detected from keywords: {[k for k in keywords if k in message_lower]}")
                    return domain

            # Check conversation history for context
            if session_state and 'conversation_history' in session_state:
                recent_messages = session_state['conversation_history'][-5:]  # Last 5 messages
                history_text = ' '.join([msg.get('content', '') for msg in recent_messages]).lower()

                for domain, keywords in domain_keywords.items():
                    keyword_count = sum(1 for keyword in keywords if keyword in history_text)
                    if keyword_count >= 2:  # If multiple keywords from same domain in recent history
                        logger.debug(f"Domain '{domain}' detected from conversation history")
                        return domain

            return "general"

        except Exception as e:
            logger.error(f"Error detecting domain context: {e}")
            return "general"

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics including adaptive RAG if available"""
        try:
            base_stats = {
                "agent_id": self.config.agent_id,
                "agent_name": self.config.name,
                "model": self.config.model,
                "created_at": self.config.created_at.isoformat() if self.config.created_at else None,
                "active_sessions": len(self.active_sessions),
                "memory_system_type": "Enhanced" if hasattr(self.memory_blossom, 'enable_adaptive_rag') else "Standard"
            }

            # Add adaptive stats if available
            if hasattr(self.memory_blossom, 'get_adaptive_stats'):
                adaptive_stats = self.memory_blossom.get_adaptive_stats()
                base_stats["adaptive_rag"] = adaptive_stats
            else:
                base_stats["adaptive_rag"] = {"enabled": False}

            # Add basic memory stats
            if hasattr(self.memory_blossom, 'memory_stores'):
                base_stats["memory_stores"] = {
                    mem_type: len(mem_list)
                    for mem_type, mem_list in self.memory_blossom.memory_stores.items()
                }
                base_stats["total_memories"] = sum(base_stats["memory_stores"].values())

            return base_stats

        except Exception as e:
            logger.error(f"Error getting enhanced stats: {e}")
            return {
                "agent_id": self.config.agent_id,
                "agent_name": self.config.name,
                "error": str(e)
            }
```

---
#### `▶ api/routes.py` (Modified)
Finally, we add an admin endpoint to create the specialist agents with the correct roles and fixed IDs.

```python
# ==================== api/routes.py
"""
Enhanced API routes with memory management, agent editing, and pre-built agent capabilities
"""

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import litellm as Litellmtop
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any, Union
import jwt
import re
from datetime import datetime, timedelta, timezone
import bcrypt
import os
from pathlib import Path
import sys
import logging
import json
import io
import zipfile
import asyncio
import uuid
import shutil
from rag_processor import process_and_index_file
from tensorart_specialist_instruction import TENSORART_SPECIALIST_INSTRUCTION
# --- MODIFIED: Import all specialist IDs ---
from agent_manager import (
    TENSORART_SPECIALIST_AGENT_ID,
    SPEECH_SPECIALIST_AGENT_ID,
    MCP_SPECIALIST_AGENT_ID,
    A2A_SPECIALIST_AGENT_ID
)
# --- END MODIFICATION ---
from ncf_processing import montar_prompt_aura_ncf

# Import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Original imports
from agent_manager import AgentManager, AgentConfig, MODEL_COSTS, NCFAuraAgentInstance
from database.models import AgentRepository, User, Agent, Memory, CreditTransaction, ChatSession, Message
from memory_system.memory_models import Memory as MemoryModel
from sqlalchemy.orm import joinedload
from sqlalchemy import desc, and_
from ncf_processing import aura_reflector_analisar_interacao, analisar_e_contribuir_para_memoria_live
from langchain_community.chat_models import litellm
from ceaf_adapter import CEAFAgentAdapter

# Imports for Pre-built Agent Extensions
from prebuilt_agents_system import (
    PrebuiltAgentRepository,
    AgentArchetype,
    AgentMaturityLevel,
    PersonalityArchitect
)

# --- NEW: Import new specialist instructions ---
from speech_specialist_instruction import SPEECH_SPECIALIST_INSTRUCTION
from mcp_specialist_instruction import MCP_SPECIALIST_INSTRUCTION
from a2a_specialist_instruction import A2A_SPECIALIST_INSTRUCTION

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ... (The rest of the file up to the admin endpoints remains the same)

# --- Find the admin endpoints section and add this new one ---

@app.post("/admin/create-specialist-agents", tags=["Admin"], response_model=dict)
async def create_all_specialist_agents(current_user: dict = Depends(verify_token)):
    """
    Creates or verifies all system-critical specialist agents (TensorArt, Speech, MCP, A2A).
    This is a required, one-time setup step for the system to have full capabilities.
    """
    # Simple admin check - in production, you'd have a proper role system.
    admin_usernames = ["xupeta", "admin"] # List of admin users
    if current_user["username"] not in admin_usernames:
        raise HTTPException(status_code=403, detail="Admin access required")

    specialists_to_create = {
        "TensorArt": {
            "id": TENSORART_SPECIALIST_AGENT_ID,
            "persona": "An AI expert in image generation using the Tensor.Art API.",
            "instruction": TENSORART_SPECIALIST_INSTRUCTION,
            "roles": ["specialist", "image_generation"]
        },
        "Speech": {
            "id": SPEECH_SPECIALIST_AGENT_ID,
            "persona": "An AI expert in text-to-speech using Hugging Face models.",
            "instruction": SPEECH_SPECIALIST_INSTRUCTION,
            "roles": ["specialist", "speech_processing"]
        },
        "MCP": {
            "id": MCP_SPECIALIST_AGENT_ID,
            "persona": "An AI expert in using tools from remote MCP servers.",
            "instruction": MCP_SPECIALIST_INSTRUCTION,
            "roles": ["specialist", "mcp_specialist"]
        },
        "A2A": {
            "id": A2A_SPECIALIST_AGENT_ID,
            "persona": "An AI expert in discovering and delegating tasks to other agents via A2A.",
            "instruction": A2A_SPECIALIST_INSTRUCTION,
            "roles": ["specialist", "a2a_specialist"]
        }
    }

    results = {}
    user_id = current_user["user_id"]

    for name, details in specialists_to_create.items():
        agent_id = details["id"]
        
        if agent_id in agent_manager._agent_configs:
            results[name] = {"status": "verified", "agent_id": agent_id}
            continue

        try:
            # Manually create the agent to ensure the correct ID and roles are set
            agent_path = agent_manager.base_storage_path / user_id / agent_id
            agent_path.mkdir(parents=True, exist_ok=True)
            
            config = AgentConfig(
                agent_id=agent_id,
                user_id=user_id,
                name=f"{name} Specialist",
                persona=details["persona"],
                detailed_persona=details["instruction"],
                model="openrouter/openai/gpt-4o-mini",
                memory_path=str(agent_path / "memory_blossom.json"), # Memory is cheap
                created_at=datetime.now(),
                settings={"system_type": "ncf", "roles": details["roles"]}
            )
            
            agent_manager._save_agent_config(config)
            agent_manager._agent_configs[agent_id] = config

            db_repo.create_agent(
                agent_id=agent_id,
                user_id=user_id,
                name=f"{name} Specialist",
                persona=details["persona"],
                detailed_persona=details["instruction"],
                model="openrouter/openai/gpt-4o-mini",
                is_public=False
            )
            results[name] = {"status": "created", "agent_id": agent_id}
        except Exception as e:
            results[name] = {"status": "error", "message": str(e)}

    return {
        "message": "Specialist agent setup process complete.",
        "results": results
    }


# ... (The rest of the file continues as before)
```
