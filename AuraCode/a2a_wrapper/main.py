# -------------------- a2a_wrapper/main.py (FIXED VERSION) --------------------

# a2a_wrapper/main.py
import uvicorn
from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import JSONResponse
import json
import uuid
from datetime import datetime, timezone
from typing import Union, Dict, Any, List, Optional
import logging
import os
from dotenv import load_dotenv
from types import SimpleNamespace

# For LLM calls within pillar/reflector functions
from google.adk.models.lite_llm import LiteLlm
from starlette.middleware.cors import CORSMiddleware
from google.adk.models.llm_request import LlmRequest

# --- Load .env file ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(PROJECT_ROOT, '.env')

if os.path.exists(dotenv_path):
    print(f"A2A Wrapper: Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    print(f"A2A Wrapper: .env file not found at {dotenv_path}. Relying on environment variables.")

# --- Module Imports ---
# orchestrator_adk_agent parts are used by this NCF-specific wrapper
from orchestrator_adk_agent import (
    adk_runner,  # The ADK runner for the NCF orchestrator agent
    ADK_APP_NAME,  # App name for the NCF orchestrator's ADK sessions
    memory_blossom_instance,  # The single MemoryBlossom for the NCF orchestrator
    AGENT_MODEL_STRING
)
from memory_system.memory_blossom import MemoryBlossom
from memory_system.memory_models import Memory as MemoryModel  # memory_system.memory_models.Memory

from a2a_wrapper.models import (
    A2APart, A2AMessage, A2ATaskSendParams, A2AArtifact,
    A2ATaskStatus, A2ATaskResult, A2AJsonRpcRequest, A2AJsonRpcResponse,
    AgentCard, AgentCardSkill, AgentCardProvider, AgentCardAuthentication, AgentCardCapabilities
)

from google.genai.types import Content as ADKContent
from google.genai.types import Part as ADKPart
from google.adk.sessions import Session as ADKSession

# FIXED: Import NCF functions with corrected signatures
from ncf_processing import (
    get_narrativa_de_fundamento_pilar1,
    get_rag_info_pilar2,
    format_chat_history_pilar3,
    montar_prompt_aura_ncf,
    aura_reflector_analisar_interacao
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

# --- Configuration & FastAPI App ---
A2A_WRAPPER_HOST = os.getenv("A2A_WRAPPER_HOST", "0.0.0.0")
A2A_WRAPPER_PORT = int(os.getenv("A2A_WRAPPER_PORT", "8094"))  # Default for this specific wrapper
A2A_WRAPPER_BASE_URL = os.getenv("A2A_WRAPPER_BASE_URL", f"http://localhost:{A2A_WRAPPER_PORT}")

app = FastAPI(
    title="Aura Agent A2A Wrapper (NCF Prototype)",
    description="Exposes the single, advanced Aura (NCF) ADK agent via the A2A protocol."
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Agent Card (Specific to the NCF Orchestrator Agent) ---
AGENT_CARD_DATA = AgentCard(
    name="AuraNCFOrchestrator",  # Distinguish from generic Aura agents
    description="The advanced Aura agent with Narrative Context Framing (NCF) capabilities, "
                "designed for deep, contextual understanding over long interactions.",
    url=A2A_WRAPPER_BASE_URL,  # This A2A server's URL
    version="1.3.0-ncf-standalone",
    provider=AgentCardProvider(organization="AuraDev", url=os.environ.get("OR_SITE_URL", "http://example.com")),
    capabilities=AgentCardCapabilities(streaming=False, pushNotifications=False),
    authentication=AgentCardAuthentication(schemes=[]),
    skills=[
        AgentCardSkill(
            id="ncf_orchestrator_conversation",
            name="Deep Narrative Conversation with Aura (NCF Orchestrator)",
            description="Engage in a sophisticated, contextual conversation with the NCF-powered Aura. "
                        "This agent uses its dedicated MemoryBlossom and advanced Narrative Context Framing.",
            tags=["chat", "conversation", "memory", "ncf", "context", "orchestrator", "advanced-ai"],
            examples=[
                "Let's explore the philosophical implications of our last discussion on AI sentience.",
                "How does the concept of 'liminality' apply to the current global situation, considering our previous chat?",
            ],
            parameters={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "The textual input from the user for the conversation."
                    },
                    "a2a_task_id_override": {
                        "type": "string",
                        "description": "Optional: Override the A2A task ID for session mapping.",
                        "nullable": True
                    }
                },
                "required": ["user_input"]
            }
        )
    ]
)


@app.get("/.well-known/agent.json", response_model=AgentCard, response_model_exclude_none=True)
async def get_agent_card_for_ncf_orchestrator():
    return AGENT_CARD_DATA


# Session mapping for this specific NCF orchestrator agent
a2a_task_to_adk_session_map: Dict[str, str] = {}
# LLM used by pillar/reflector functions (distinct from the main agent's LLM if needed, but can be the same)
helper_llm = LiteLlm(model=AGENT_MODEL_STRING)


# --- A2A RPC Handler (for the NCF Orchestrator Agent) ---
@app.post("/", response_model=A2AJsonRpcResponse, response_model_exclude_none=True)
async def handle_ncf_orchestrator_a2a_rpc(rpc_request: A2AJsonRpcRequest, http_request: FastAPIRequest):
    client_host = http_request.client.host if http_request.client else "unknown_host"
    logger.info(
        f"\nNCF A2A Wrapper: Request from {client_host}: Method={rpc_request.method}, RPC_ID={rpc_request.id}")

    if rpc_request.method == "tasks/send":
        if rpc_request.params is None:
            logger.error("NCF A2A Wrapper: Missing 'params' for tasks/send")
            return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32602, "message": "Invalid params: missing"})

        try:
            task_params = rpc_request.params
            logger.info(f"NCF A2A Wrapper: Processing tasks/send for A2A Task ID: {task_params.id}")

            user_utterance_raw = ""
            if task_params.message and task_params.message.parts:
                first_part = task_params.message.parts[0]
                if first_part.type == "data" and first_part.data and "user_input" in first_part.data:
                    user_utterance_raw = first_part.data["user_input"]
                elif first_part.type == "text" and first_part.text:  # Fallback if 'data' with 'user_input' not found
                    user_utterance_raw = first_part.text

            if not user_utterance_raw:
                logger.error("NCF A2A Wrapper: No user_input or text found in A2A message.")
                return A2AJsonRpcResponse(id=rpc_request.id,
                                          error={"code": -32602, "message": "Invalid params: user_input/text missing"})

            adk_session_key_override = None
            if task_params.message.parts[0].data and task_params.message.parts[0].data.get("a2a_task_id_override"):
                adk_session_key_override = task_params.message.parts[0].data["a2a_task_id_override"]

            # Use the NCF Orchestrator's ADK_APP_NAME and its adk_runner
            # The adk_runner is imported from orchestrator_adk_agent
            ncf_adk_app_name = ADK_APP_NAME  # From orchestrator_adk_agent
            ncf_adk_runner = adk_runner  # From orchestrator_adk_agent

            adk_session_map_key = adk_session_key_override or task_params.sessionId or task_params.id
            adk_user_id = f"ncf_a2a_user_{adk_session_map_key}"  # Distinguish user for NCF context
            adk_session_id = a2a_task_to_adk_session_map.get(adk_session_map_key)

            current_adk_session: Optional[ADKSession] = None
            if adk_session_id:
                try:
                    current_adk_session = ncf_adk_runner.session_service.get_session(
                        app_name=ncf_adk_app_name, user_id=adk_user_id, session_id=adk_session_id
                    )
                    if not current_adk_session:
                        logger.warning(f"NCF A2A: get_session returned None for {adk_session_id}")
                except Exception as e_get:
                    logger.error(f"NCF A2A: Error get_session: {e_get}", exc_info=True)

            if not current_adk_session:
                adk_session_id = f"ncf_adk_session_{adk_session_map_key}_{str(uuid.uuid4())[:8]}"
                a2a_task_to_adk_session_map[adk_session_map_key] = adk_session_id
                current_adk_session = ncf_adk_runner.session_service.create_session(
                    app_name=ncf_adk_app_name, user_id=adk_user_id, session_id=adk_session_id,
                    state={'conversation_history': [], 'foundation_narrative_turn_count': 0}
                )
                logger.info(f"NCF A2A Wrapper: Created ADK session '{adk_session_id}' for user '{adk_user_id}'")
            else:
                logger.info(f"NCF A2A Wrapper: Reusing ADK session '{adk_session_id}' for user '{adk_user_id}'")

            if 'conversation_history' not in current_adk_session.state:
                current_adk_session.state['conversation_history'] = []
            if 'foundation_narrative_turn_count' not in current_adk_session.state:
                current_adk_session.state['foundation_narrative_turn_count'] = 0

            current_adk_session.state['conversation_history'].append({"role": "user", "content": user_utterance_raw})

            # FIXED: Use corrected NCF function signatures
            # Use the NCF orchestrator's memory_blossom_instance
            narrativa_fundamento = await get_narrativa_de_fundamento_pilar1(
                session_state=current_adk_session.state,
                memory_blossom=memory_blossom_instance,
                user_id=adk_user_id,
                llm_instance=helper_llm,
                agent_name="Aura (NCF Orchestrator)",
                agent_persona="IA conversacional avançada com capacidades de NCF"
            )

            rag_info_list = await get_rag_info_pilar2(
                user_utterance=user_utterance_raw,
                memory_blossom=memory_blossom_instance,
                session_state=current_adk_session.state
            )

            chat_history_for_prompt_str = format_chat_history_pilar3(
                chat_history_list=current_adk_session.state['conversation_history']
            )

            # Persona details for the NCF orchestrator
            ncf_aura_persona_agente = "Aura (NCF Orchestrator)"
            ncf_aura_persona_detalhada = "Você é uma IA conversacional avançada, com capacidades de NCF (Narrative Context Framing). Você mantém uma compreensão profunda e contextual das conversas, formando memórias e construindo narrativas coerentes ao longo do tempo. Always reply in the same language the user is using."

            final_ncf_prompt_str = montar_prompt_aura_ncf(
                agent_name=ncf_aura_persona_agente,
                agent_detailed_persona=ncf_aura_persona_detalhada,
                narrativa_fundamento=narrativa_fundamento,
                informacoes_rag_list=rag_info_list,
                chat_history_recente_str=chat_history_for_prompt_str,
                user_reply=user_utterance_raw
            )

            adk_input_content = ADKContent(role="user", parts=[ADKPart(text=final_ncf_prompt_str)])
            logger.info(f"NCF A2A Wrapper: Running NCF Aura ADK agent for session '{adk_session_id}'")
            adk_agent_final_text_response = None

            # Update session state in service before running agent
            # This ensures tool_context gets the latest history
            current_adk_session_for_update = ncf_adk_runner.session_service.get_session(
                app_name=ncf_adk_app_name, user_id=adk_user_id, session_id=adk_session_id
            )
            if current_adk_session_for_update:
                current_adk_session_for_update.state = current_adk_session.state  # update with latest history
                ncf_adk_runner.session_service.update_session(current_adk_session_for_update)
            else:
                logger.error(
                    f"NCF A2A: Session not found before ADK run for {adk_session_id}. History for tools might be stale.")

            async for event in ncf_adk_runner.run_async(
                    user_id=adk_user_id, session_id=adk_session_id, new_message=adk_input_content
            ):
                # Event logging
                if event.get_function_calls():
                    fc = event.get_function_calls()[0]
                    logger.info(f"    NCF ADK FuncCall: {fc.name}({json.dumps(fc.args)})")
                if event.get_function_responses():
                    fr = event.get_function_responses()[0]
                    logger.info(f"    NCF ADK FuncResp: {fr.name} -> {str(fr.response)[:100]}...")

                if event.is_final_response():
                    if event.content and event.content.parts and event.content.parts[0].text:
                        adk_agent_final_text_response = event.content.parts[0].text.strip()
                    break

            adk_agent_final_text_response = adk_agent_final_text_response or "(Aura NCF não forneceu uma resposta)"
            current_adk_session.state['conversation_history'].append(
                {"role": "assistant", "content": adk_agent_final_text_response})

            # FIXED: Use corrected reflector function signature
            await aura_reflector_analisar_interacao(
                user_utterance=user_utterance_raw,
                prompt_ncf_usado=final_ncf_prompt_str,
                resposta_de_aura=adk_agent_final_text_response,
                memory_blossom=memory_blossom_instance,
                user_id=adk_user_id,
                llm_instance=helper_llm
            )

            # Persist final state
            final_adk_session_state = ncf_adk_runner.session_service.get_session(
                app_name=ncf_adk_app_name, user_id=adk_user_id, session_id=adk_session_id
            )
            if final_adk_session_state:
                final_adk_session_state.state = current_adk_session.state  # Ensure it has the latest history
                ncf_adk_runner.session_service.update_session(final_adk_session_state)

            a2a_response_artifact = A2AArtifact(parts=[A2APart(type="text", text=adk_agent_final_text_response)])
            a2a_task_status = A2ATaskStatus(state="completed")
            a2a_task_result = A2ATaskResult(
                id=task_params.id, sessionId=task_params.sessionId,
                status=a2a_task_status, artifacts=[a2a_response_artifact]
            )
            logger.info(f"NCF A2A Wrapper: Sending A2A response for Task ID {task_params.id}")
            return A2AJsonRpcResponse(id=rpc_request.id, result=a2a_task_result)

        except ValueError as ve:  # Catch pydantic validation errors
            logger.error(f"NCF A2A Wrapper: Value Error (likely Pydantic): {ve}", exc_info=True)
            return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32602, "message": f"Invalid params: {ve}"})
        except Exception as e:
            logger.error(f"NCF A2A Wrapper: Internal Error: {e}", exc_info=True)
            return A2AJsonRpcResponse(id=rpc_request.id,
                                      error={"code": -32000, "message": f"Internal Server Error: {e}"})
    else:
        logger.warning(f"NCF A2A Wrapper: Method '{rpc_request.method}' not supported.")
        return A2AJsonRpcResponse(id=rpc_request.id,
                                  error={"code": -32601, "message": f"Method not found: {rpc_request.method}"})


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: OPENROUTER_API_KEY is not set. Agent will likely fail LLM calls. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.info(f"Starting Aura NCF Orchestrator A2A Wrapper Server on {A2A_WRAPPER_HOST}:{A2A_WRAPPER_PORT}")
    logger.info(f"NCF Agent Card available at: {A2A_WRAPPER_BASE_URL}/.well-known/agent.json")
    # Make sure to run this specific FastAPI app
    uvicorn.run("a2a_wrapper.main:app", host=A2A_WRAPPER_HOST, port=A2A_WRAPPER_PORT, reload=True)