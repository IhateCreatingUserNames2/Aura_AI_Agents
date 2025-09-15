# ==================== a2a_wrapper/multi_agent_a2a_wrapper.py ====================
"""
Updated multi_agent_a2a_wrapper.py to work with NCF-enabled agents
"""

from fastapi import FastAPI, HTTPException
from a2a_wrapper.models import *
from agent_manager import AgentManager  # Now NCF-enabled
import json
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Aura A2A Service (NCF-Enabled)")
agent_manager = AgentManager()

# Store active agent card registrations
registered_agents = {}


@app.post("/agents/{agent_id}/register")
async def register_agent_a2a(agent_id: str):
    """Register a specific NCF-enabled agent for A2A access"""
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance:
        raise HTTPException(status_code=404, detail="Agent not found")

    config = agent_instance.config

    # Create agent-specific A2A card with NCF capabilities
    agent_card = AgentCard(
        name=config.name,
        description=f"{config.persona}. NCF-Enabled with advanced memory and contextual understanding. {config.detailed_persona[:200]}...",
        url=f"http://localhost:8000/agents/{agent_id}",  # Adjust base URL as needed
        version="2.0.0",
        provider=AgentCardProvider(
            organization="User Created (NCF-Enabled)",
            url="http://localhost:8000"
        ),
        capabilities=AgentCardCapabilities(streaming=False),
        authentication=AgentCardAuthentication(schemes=[]),
        skills=[
            AgentCardSkill(
                id="ncf_conversation",
                name=f"Advanced Chat with {config.name} (NCF)",
                description=f"Have an advanced conversation with {config.name}. This agent features Narrative Context Framing (NCF), isolated memory system, RAG capabilities, and reflective analysis. {config.persona}",
                tags=["ncf", "memory", "conversation", "rag", "reflector", "advanced"],
                examples=[
                    f"Let's explore a complex topic with {config.name}'s advanced understanding",
                    f"Tell {config.name} about your long-term goals for contextual memory",
                    f"Discuss philosophical concepts with {config.name}'s narrative foundation"
                ],
                parameters={
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "Your message to the NCF-enabled agent"
                        },
                        "context_aware": {
                            "type": "boolean",
                            "description": "Enable advanced NCF context processing (recommended: true)",
                            "default": True
                        }
                    },
                    "required": ["user_input"]
                }
            )
        ]
    )

    registered_agents[agent_id] = agent_card
    return {
        "status": "registered",
        "agent_id": agent_id,
        "ncf_enabled": True,
        "capabilities": ["ncf", "memory", "rag", "reflector"]
    }


@app.get("/agents/{agent_id}/.well-known/agent.json")
async def get_agent_card(agent_id: str):
    """Get A2A agent card for specific NCF-enabled agent"""
    if agent_id not in registered_agents:
        raise HTTPException(status_code=404, detail="Agent not registered for A2A")
    return registered_agents[agent_id]


@app.post("/agents/{agent_id}")
async def handle_agent_a2a_rpc(agent_id: str, rpc_request: A2AJsonRpcRequest):
    """Handle A2A RPC requests for specific NCF-enabled agent"""

    if rpc_request.method != "tasks/send":
        return A2AJsonRpcResponse(
            id=rpc_request.id,
            error={"code": -32601, "message": "Method not found"}
        )

    # Get NCF-enabled agent instance
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance:
        return A2AJsonRpcResponse(
            id=rpc_request.id,
            error={"code": -32602, "message": "NCF-enabled agent not found"}
        )

    try:
        # Extract user input
        task_params = rpc_request.params
        user_input = ""

        if task_params.message and task_params.message.parts:
            first_part = task_params.message.parts[0]
            if first_part.type == "data" and first_part.data:
                user_input = first_part.data.get("user_input", "")
            elif first_part.type == "text":
                user_input = first_part.text

        if not user_input:
            return A2AJsonRpcResponse(
                id=rpc_request.id,
                error={"code": -32602, "message": "No user input found"}
            )

        # Process with NCF-enabled agent (full NCF capabilities automatically applied)
        response = await agent_instance.process_message(
            user_id=f"a2a_{task_params.id}",
            session_id=task_params.sessionId or task_params.id,
            message=user_input
        )

        # Build A2A response with NCF metadata
        result = A2ATaskResult(
            id=task_params.id,
            sessionId=task_params.sessionId,
            status=A2ATaskStatus(state="completed"),
            artifacts=[
                A2AArtifact(
                    name="NCF Response",
                    description="Response from NCF-enabled Aura agent with contextual understanding",
                    parts=[A2APart(type="text", text=response)],
                    metadata={
                        "ncf_enabled": True,
                        "agent_name": agent_instance.config.name,
                        "capabilities": ["narrative_foundation", "rag", "reflector", "memory"]
                    }
                )
            ],
            metadata={
                "ncf_processed": True,
                "agent_capabilities": ["ncf", "memory", "rag", "reflector"]
            }
        )

        return A2AJsonRpcResponse(id=rpc_request.id, result=result)

    except Exception as e:
        logger.error(f"Error processing A2A request for NCF agent: {e}")
        return A2AJsonRpcResponse(
            id=rpc_request.id,
            error={"code": -32000, "message": f"NCF agent processing error: {str(e)}"}
        )


# Integration with AIRA Hub for NCF agents
@app.post("/register-to-hub/{agent_id}")
async def register_to_aira_hub(agent_id: str, hub_url: str):
    """Register a specific NCF-enabled agent to AIRA Hub"""
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance:
        raise HTTPException(status_code=404, detail="Agent not found")

    config = agent_instance.config

    # Prepare registration data for AIRA Hub with NCF capabilities
    registration_data = {
        "url": f"http://localhost:8000/agents/{agent_id}",  # Adjust base URL as needed
        "name": f"{config.name}_NCF_A2A",
        "description": f"NCF-Enabled {config.persona}. Advanced contextual AI with memory, narrative foundation, and reflective capabilities.",
        "version": "2.0.0",
        "mcp_tools": [
            {
                "name": f"TalkTo_{config.name.replace(' ', '_')}_NCF",
                "description": f"Advanced chat with {config.name} (NCF-enabled). Features narrative foundation, RAG, memory system, and reflective analysis. {config.persona}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "Your message to the NCF-enabled agent"
                        },
                        "enable_ncf": {
                            "type": "boolean",
                            "description": "Enable full NCF processing (recommended: true)",
                            "default": True
                        }
                    },
                    "required": ["user_input"]
                },
                "annotations": {
                    "aira_bridge_type": "a2a",
                    "aira_a2a_target_skill_id": "ncf_conversation",
                    "aira_a2a_agent_url": f"http://localhost:8000/agents/{agent_id}",
                    "ncf_enabled": True,
                    "capabilities": ["narrative_foundation", "rag", "reflector", "memory", "contextual_understanding"]
                }
            }
        ],
        "a2a_skills": [],
        "aira_capabilities": ["a2a", "ncf", "memory", "contextual_ai"],
        "status": "online",
        "tags": ["user-created", "aura", "a2a", "ncf", "memory", "advanced-ai"],
        "category": "NCF_UserAgents",
        "ncf_metadata": {
            "narrative_foundation": True,
            "rag_enabled": True,
            "reflector_analysis": True,
            "isolated_memory": True,
            "model": config.model
        }
    }

    # Make registration request to AIRA Hub
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{hub_url}/register", json=registration_data)
        if response.status_code == 200:
            return {
                "status": "registered",
                "hub_response": response.json(),
                "ncf_enabled": True,
                "capabilities": ["ncf", "memory", "rag", "reflector"]
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Hub registration failed: {response.text}"
            )