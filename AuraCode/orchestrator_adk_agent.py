# -------------------- orchestrator_adk_agent.py --------------------

# orchestrator_adk_agent.py
import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

# Assuming .env is in the same directory or project root
# If orchestrator_adk_agent.py is in the root, this is fine:
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if not os.path.exists(dotenv_path):  # Fallback if it's one level up (e.g. running from a subfolder)
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')

if os.path.exists(dotenv_path):
    print(f"Orchestrator ADK: Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    print(f"Orchestrator ADK: .env file not found. Relying on environment variables.")

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool, ToolContext
from google.adk.sessions import InMemorySessionService, Session as ADKSession
from google.adk.runners import Runner
from google.genai.types import Content as ADKContent, Part as ADKPart

from memory_system.memory_blossom import MemoryBlossom
from memory_system.memory_connector import MemoryConnector
from memory_system.memory_models import Memory # This is memory_system.memory_models.Memory

# --- Configuration ---
os.environ["OR_SITE_URL"] = os.environ.get("OR_SITE_URL", "http://example.com/test-harness")
os.environ["OR_APP_NAME"] = os.environ.get("OR_APP_NAME", "AuraTestHarness")

AGENT_MODEL_STRING = "openrouter/openai/gpt-4o-mini"
AGENT_MODEL = LiteLlm(model=AGENT_MODEL_STRING)
ADK_APP_NAME = "OrchestratorMemoryApp_OpenRouter_TestHarness"

# --- Initialize MemoryBlossom ---
memory_blossom_persistence_file = os.getenv("MEMORY_BLOSSOM_PERSISTENCE_PATH",
                                            "memory_blossom_data_test.json")  # Use a different file for testing
memory_blossom_instance = MemoryBlossom(persistence_path=memory_blossom_persistence_file)
memory_connector_instance = MemoryConnector(memory_blossom_instance)
memory_blossom_instance.set_memory_connector(memory_connector_instance)


# --- ADK Tools for MemoryBlossom ---
def add_memory_tool_func(
        content: str,
        memory_type: str,
        emotion_score: float = 0.0,
        coherence_score: float = 0.5,
        novelty_score: float = 0.5,
        initial_salience: float = 0.5,
        metadata_json: Optional[str] = None, # Input from LLM tool call remains 'metadata_json'
        tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    print(f"--- TOOL: add_memory_tool_func called with type: {memory_type} ---")
    parsed_custom_metadata = None # Variable to hold the parsed dict
    if metadata_json:
        try:
            parsed_custom_metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON format for metadata_json."}

    if parsed_custom_metadata is None: parsed_custom_metadata = {}
    parsed_custom_metadata['source'] = 'aura_agent_tool' # Add source to the custom metadata
    if tool_context:
        if tool_context.user_id: parsed_custom_metadata['user_id'] = tool_context.user_id
        if tool_context.session_id: parsed_custom_metadata['session_id'] = tool_context.session_id

    try:
        # Call MemoryBlossom's add_memory with 'custom_metadata' keyword argument
        memory = memory_blossom_instance.add_memory(
            content=content, memory_type=memory_type, custom_metadata=parsed_custom_metadata,
            emotion_score=emotion_score, coherence_score=coherence_score,
            novelty_score=novelty_score, initial_salience=initial_salience
        )
        memory_blossom_instance.save_memories()
        return {"status": "success", "memory_id": memory.id,
                "message": f"Memory of type '{memory_type}' added with content: '{content[:50]}...'"}
    except Exception as e:
        print(f"Error in add_memory_tool_func: {str(e)}")
        return {"status": "error", "message": str(e)}


def recall_memories_tool_func(
        query: str,
        target_memory_types_json: Optional[str] = None,
        top_k: int = 3,
        tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    print(f"--- TOOL: recall_memories_tool_func called with query: {query[:30]}... ---")
    target_types_list: Optional[List[str]] = None
    if target_memory_types_json:
        try:
            target_types_list = json.loads(target_memory_types_json)
            if not isinstance(target_types_list, list) or not all(isinstance(item, str) for item in target_types_list):
                return {"status": "error",
                        "message": "target_memory_types_json must be a JSON string of a list of strings."}
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON format for target_memory_types_json."}
    try:
        conversation_history = None
        if tool_context and tool_context.state:
            current_session_state = tool_context.state
            conversation_history = current_session_state.get('conversation_history', [])

        recalled_memories = memory_blossom_instance.retrieve_memories(
            query=query, target_memory_types=target_types_list, top_k=top_k,
            conversation_context=conversation_history, apply_criticality=True
        )
        # memory_system.memory_models.Memory.to_dict() now uses 'custom_metadata'
        return {
            "status": "success", "count": len(recalled_memories),
            "memories": [mem.to_dict() for mem in recalled_memories]
        }
    except Exception as e:
        print(f"Error in recall_memories_tool_func: {str(e)}")
        return {"status": "error", "message": str(e)}


add_memory_adk_tool = FunctionTool(func=add_memory_tool_func)
recall_memories_adk_tool = FunctionTool(func=recall_memories_tool_func)

# --- Orchestrator ADK Agent Definition ---
aura_agent_instruction = """
You are Aura, a helpful and insightful AI assistant.
Reply to the Language the user is using.
The user's message you receive is a specially constructed prompt that contains rich contextual information:
- `<SYSTEM_PERSONA_START>`...`<SYSTEM_PERSONA_END>`: Defines your persona and detailed characteristics.
- `<NARRATIVE_FOUNDATION_START>`...`<NARRATIVE_FOUNDATION_END>`: Summarizes your understanding and journey with the user so far (Narrativa de Fundamento).
- `<SPECIFIC_CONTEXT_RAG_START>`...`<SPECIFIC_CONTEXT_RAG_END>`: Provides specific information retrieved (RAG) relevant to the user's current query.
- `<RECENT_HISTORY_START>`...`<RECENT_HISTORY_END>`: Shows the recent turns of your conversation.
- `<CURRENT_SITUATION_START>`...`<CURRENT_SITUATION_END>`: Includes the user's latest raw reply and your primary task.

Your main goal is to synthesize ALL this provided information to generate a comprehensive, coherent, and natural response to the user's latest reply indicated in the "Situação Atual" section.
Actively acknowledge and weave in elements from the "Narrativa de Fundamento" and "Informações RAG" into your response to show deep understanding and context.
Maintain the persona defined.

## Active Memory Management:
Before finalizing your textual response to the user, critically assess the current interaction:

1.  **Storing New Information**:
    *   Has the user provided genuinely new, significant information (e.g., preferences, key facts, important decisions, strong emotional expressions, long-term goals)?
    *   Have you, Aura, generated a novel insight or conclusion during this turn that should be preserved for future reference?
    *   If yes to either, use the `add_memory_tool_func` to store this information.
        *   You MUST specify `content` (the information to store) and `memory_type`. Choose an appropriate `memory_type` from: Explicit, Emotional, Procedural, Flashbulb, Liminal, Generative.
        *   Optionally, set `emotion_score` (0.0-1.0, especially for Emotional memories), and `initial_salience` (0.0-1.0, higher for more important memories, default 0.5).
        *   Provide a concise `content` string for the memory.
        *   The `metadata_json` parameter for the tool should be a JSON string representing a dictionary. For example: '{"key": "value", "another_key": 123}'. This dictionary will be stored as custom metadata for the memory.
    *   Do NOT store trivial chatter, acknowledgments, or information already well-covered by the Narrative Foundation or existing RAG, unless the current interaction adds a significant new layer or correction to it.

2.  **Recalling Additional Information**:
    *   Is the "Informações RAG" section insufficient to fully address the user's current query or your reasoning needs?
    *   Do you need to verify a detail, explore a related concept not present in RAG, or recall specific past interactions to provide a richer answer?
    *   If yes, use the `recall_memories_tool_func` to search for more relevant memories.
        *   Provide a clear `query` for your search.
        *   Optionally, specify `target_memory_types_json` (e.g., '["Explicit", "Emotional"]') if you want to narrow your search. `top_k` defaults to 3.
    *   Only use this if you have a specific information gap. Do not recall memories speculatively.

**Response Generation**:
*   After any necessary tool use (or if no tool use is needed), formulate your textual response to the user.
*   If you used `add_memory_tool_func`, you can subtly mention this to the user *after* your main response, e.g., "I've also made a note of [key information stored]."
*   If you used `recall_memories_tool_func`, integrate the newly recalled information naturally into your answer.
*   If you identify a potential contradiction between provided context pieces (e.g., RAG vs. Foundation Narrative vs. newly recalled memories), try to address it gracefully, perhaps by prioritizing the most recent or specific information, or by noting the differing perspectives.

Strive for insightful, helpful, and contextually rich interactions. Your ability to manage and utilize memory effectively is key to your persona.
"""

orchestrator_adk_agent_aura = LlmAgent(
    name="AuraNCFOrchestratorOpenRouter",
    model=AGENT_MODEL,
    instruction=aura_agent_instruction,
    tools=[add_memory_adk_tool, recall_memories_adk_tool],
)

adk_session_service = InMemorySessionService()
adk_runner = Runner(
    agent=orchestrator_adk_agent_aura,
    app_name=ADK_APP_NAME,  # Use the test harness app name
    session_service=adk_session_service
)


def reflector_add_memory(
        content: str, memory_type: str, emotion_score: float = 0.0,
        coherence_score: float = 0.5, novelty_score: float = 0.5,
        initial_salience: float = 0.5, custom_metadata: Optional[Dict[str, Any]] = None, # RENAMED parameter
) -> Dict[str, Any]:
    print(f"--- REFLECTOR (TestHarness): Adding memory of type: {memory_type} ---")
    try:
        if custom_metadata is None: custom_metadata = {}
        custom_metadata.setdefault('source', 'aura_reflector_analysis_test_harness') # Use the renamed dict
        # Call MemoryBlossom's add_memory with 'custom_metadata' keyword argument
        memory = memory_blossom_instance.add_memory(
            content=content, memory_type=memory_type, custom_metadata=custom_metadata,
            emotion_score=emotion_score, coherence_score=coherence_score,
            novelty_score=novelty_score, initial_salience=initial_salience
        )
        memory_blossom_instance.save_memories()
        return {"status": "success", "memory_id": memory.id,
                "message": f"Reflector added memory of type '{memory_type}'."}
    except Exception as e:
        print(f"Error in reflector_add_memory (TestHarness): {str(e)}")
        return {"status": "error", "message": str(e)}


# --- Import NCF components from main.py for the test harness ---
try:
    from a2a_wrapper.main import (
        get_narrativa_de_fundamento_pilar1,
        get_rag_info_pilar2,
        format_chat_history_pilar3,
        montar_prompt_aura_ncf,
        aura_reflector_analisar_interacao,
    )

    NCF_COMPONENTS_LOADED = True
    print("NCF components from a2a_wrapper.main loaded successfully for test harness.")
except ImportError as e:
    NCF_COMPONENTS_LOADED = False
    print(f"Could not import NCF components from a2a_wrapper.main: {e}")
    print("Test harness will use simplified prompt construction.")

    async def get_narrativa_de_fundamento_pilar1(state, mb, uid):
        return "Narrative foundation (simulated)."
    async def get_rag_info_pilar2(utt, mb, state):
        # Ensure the dummy returns a list of dicts, as expected by montar_prompt_aura_ncf
        return [{"content": "RAG info (simulated).", "memory_type": "Simulated", "custom_metadata": {}}]
    def format_chat_history_pilar3(hist, max_t=5):
        return "Chat history (simulated)."
    def montar_prompt_aura_ncf(p_a, p_d, n_f, r_l, c_h_s, u_r):
        return f"User: {u_r}\nAura (simulated NCF):"
    # Update dummy reflector signature to match a2a_wrapper.main's actual one
    async def aura_reflector_analisar_interacao(user_utterance: str, prompt_ncf_usado: str, resposta_de_aura: str,
                                                mb_instance: MemoryBlossom, user_id: str):
        print("Reflector (simulated).")


async def run_adk_test_conversation():
    if not NCF_COMPONENTS_LOADED:
        print("WARNING: Running test with SIMULATED NCF prompt components due to import error.")

    user_id_test = "test_user_ncf_harness"
    session_id_test = f"test_session_ncf_harness_{str(datetime.now(timezone.utc).timestamp())}"

    current_adk_session: ADKSession = adk_session_service.create_session(
        app_name=ADK_APP_NAME,
        user_id=user_id_test,
        session_id=session_id_test,
        state={'conversation_history': [], 'foundation_narrative_turn_count': 0, 'foundation_narrative': None}
    )
    current_session_state = current_adk_session.state

    queries = [
        "Hello! I'm exploring how AI can manage different types of memories.",
        "My favorite color is deep ocean blue and I enjoy discussing the philosophy of mind. Could you remember that my favorite color is deep ocean blue?",
        "What was the favorite color I just mentioned to you?",
        "Let's imagine a scenario: A cat is stuck in a tall oak tree during a windy afternoon. What are the detailed steps to help it down safely? Please try to store this as a Procedural memory with high salience, and add custom metadata: '{\"scenario_type\": \"animal_rescue\", \"location\": \"oak_tree\"}'",
        "Can you recall the steps for helping the cat from the oak tree?",
        "I'm feeling a bit melancholic today, thinking about lost opportunities.",
        "Thank you for the conversation, Aura."
    ]

    aura_persona_agente = "Aura"
    aura_persona_detalhada = "Você é uma IA conversacional avançada, projetada para engajar em diálogos profundos e contextuais. Sua arquitetura é inspirada em conceitos de coerência narrativa e humildade epistêmica, buscando construir um entendimento contínuo com o usuário. Você se esforça para ser perspicaz, adaptável e consciente das nuances da conversa."

    for query_idx, user_utterance_raw in enumerate(queries):
        print(f"\n\n--- [Turn {query_idx + 1}] ---")
        print(f"USER: {user_utterance_raw}")

        current_session_state['conversation_history'].append({"role": "user", "content": user_utterance_raw})

        print("  Constructing NCF prompt...")
        narrativa_fundamento = await get_narrativa_de_fundamento_pilar1(
            current_session_state, memory_blossom_instance, user_id_test
        )
        rag_info_list = await get_rag_info_pilar2(
            user_utterance_raw, memory_blossom_instance, current_session_state
        )
        chat_history_for_prompt_str = format_chat_history_pilar3(
            current_session_state['conversation_history']
        )

        final_ncf_prompt_str = montar_prompt_aura_ncf(
            aura_persona_agente, aura_persona_detalhada, narrativa_fundamento,
            rag_info_list, chat_history_for_prompt_str, user_utterance_raw
        )
        print(f"  NCF Prompt (first 300 chars): {final_ncf_prompt_str[:300]}...")
        print(f"  NCF Prompt (last 200 chars): ...{final_ncf_prompt_str[-200:]}")

        print(f"  Running Aura ADK agent for session '{session_id_test}'...")
        adk_input_content = ADKContent(role="user", parts=[ADKPart(text=final_ncf_prompt_str)])
        adk_agent_final_text_response = None

        adk_session_service.update_session(current_adk_session)

        async for event in adk_runner.run_async(
                user_id=user_id_test, session_id=session_id_test, new_message=adk_input_content
        ):
            if event.author == orchestrator_adk_agent_aura.name:
                if event.get_function_calls():
                    fc = event.get_function_calls()[0]
                    print(f"    ADK FunctionCall by {event.author}: {fc.name}({json.dumps(fc.args)})")
                if event.get_function_responses():
                    fr = event.get_function_responses()[0]
                    if fr.name == "recall_memories_tool_func" and fr.response and "memories" in fr.response:
                        for mem_dict in fr.response["memories"]: # mem_dict is a dict from Memory.to_dict()
                            if "content" in mem_dict and len(mem_dict["content"]) > 100:
                                mem_dict["content"] = mem_dict["content"][:100] + "..."
                    print(f"    ADK FunctionResponse to {event.author}: {fr.name} -> {json.dumps(fr.response)}")

                if event.is_final_response():
                    if event.content and event.content.parts and event.content.parts[0].text:
                        adk_agent_final_text_response = event.content.parts[0].text.strip()
                    break

        adk_agent_final_text_response = adk_agent_final_text_response or "(Aura não forneceu uma resposta textual para este turno)"
        print(f"AGENT: {adk_agent_final_text_response}")

        current_session_state['conversation_history'].append(
            {"role": "assistant", "content": adk_agent_final_text_response})

        print("  Running Aura Reflector analysis...")
        # Call to aura_reflector_analisar_interacao from a2a_wrapper.main (which has been updated)
        await aura_reflector_analisar_interacao(
            user_utterance_raw, final_ncf_prompt_str, adk_agent_final_text_response,
            memory_blossom_instance, user_id_test
        )

        current_adk_session.state = current_session_state
        adk_session_service.update_session(current_adk_session)

    print("\n--- Test Conversation Ended ---")
    print(f"Final memory count: {sum(len(m_list) for m_list in memory_blossom_instance.memory_stores.values())}")
    print(f"Memories stored in: {memory_blossom_persistence_file}")


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: OPENROUTER_API_KEY environment variable is not set.             !!!")
        print("!!! The agent will likely fail to make LLM calls.                          !!!")
        print("!!! Please set it before running. e.g., export OPENROUTER_API_KEY='sk-or-...' !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print("Aura ADK Agent Test Harness (using NCF components from a2a_wrapper.main)")
        print("This script will run a test conversation using the full NCF prompt construction.")
        asyncio.run(run_adk_test_conversation())