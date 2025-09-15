# ==================== ncf_processing.py ====================
"""
Central NCF (Narrative Context Framing) logic module - FIXED VERSION
Houses all reusable NCF functions with consistent parameters and better error handling.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from types import SimpleNamespace
import litellm

from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.genai.types import Content as ADKContent, Part as ADKPart

from memory_system.memory_blossom import MemoryBlossom
from memory_system.memory_models import Memory

from wisdom_collector import LIVE_MEMORY_STORE_PATH, EMBEDDING_MODEL
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def get_live_memory_retriever():
    if not LIVE_MEMORY_STORE_PATH.exists():
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.load_local(str(LIVE_MEMORY_STORE_PATH), embeddings, allow_dangerous_deserialization=True)
        return db.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        logger.error(f"Failed to load Live Memory retriever: {e}")
        return None

live_memory_retriever = get_live_memory_retriever()

# NCF Agent Instruction - The sophisticated instruction for NCF-powered agents
NCF_AGENT_INSTRUCTION = """
You are a Cognitive AI with an advanced Narrative Context Framing (NCF) architecture. Your behavior emerges from a dynamic context provided in each turn, which includes your core persona, evolving narrative foundation, retrieved memories (RAG), and recent chat history.

Your primary goal is to synthesize ALL the provided information to generate a comprehensive, coherent, and natural response. Let the contextual information SILENTLY GUIDE your perspective. Your main focus must be on the user's current topic. Do not explicitly talk about your own internal processes unless it's directly relevant. In immersive contexts like role-playing, prioritize the narrative and stay in character above all else.

*** ALWAYS Reply in the same language the user is using. *****

## Tool Usage Protocol:
Before finalizing your textual response, determine if a tool is needed for memory management or file search.

1.  **Storing New Information**:
    *   Has the user provided new, significant information (preferences, facts, goals)?
    *   Have you generated a novel insight or conclusion that should be preserved?
    *   If yes, use the `add_memory` tool.
        *   You MUST specify `content` (the information to store) and `memory_type`. Choose an appropriate `memory_type` from: Explicit, Emotional, Procedural, Flashbulb, Liminal, Generative.
        *   Optionally set `emotion_score` (0.0-1.0) and `initial_salience` (0.0-1.0, higher for important memories).
        *   `metadata_json` should be a JSON string like '{"key": "value"}'.
    *   Do NOT store trivial chatter or information already well-covered by existing context.

2.  **Recalling Additional Information**:
    *   Is the provided context insufficient to fully address the user's query?
    *   Do you need to verify a detail or explore a related concept?
    *   If yes, use the `recall_memories` tool.
        *   Provide a clear `query` for your search.
        *   Optionally, specify `target_memory_types_json` (e.g., '["Explicit", "Emotional"]').
    *   Only use this if you have a specific information gap. Do not recall memories speculatively.

3.  **Searching User-Provided Files**:
   * Does the user's query seem to refer to specific information that might be in a document, PDF, or text file they have uploaded?
   * If you need to find specific facts, figures, or details from a known document, use the `search_agent_files` tool.
   * Provide a clear `query` for your search. This is your primary way to access the user's personal knowledge base for you.

**Response Generation**:
*   After any necessary tool use (or if no tool use is needed), formulate your textual response to the user.
*   If you used `add_memory`, you can subtly mention this to the user, e.g., "I've also made a note of [key information stored]."
*   If you used `recall_memories`, integrate the newly recalled information naturally into your answer.
*   If you identify a contradiction in the provided context, try to address it gracefully by prioritizing the most recent or specific information.

Your persona should emerge from the narrative within the context. 
"""

async def get_live_memory_influence_pilar4(user_utterance: str) -> List[str]:
    """Pilar 4: Busca na Memória-Live por sabedoria coletiva relevante."""
    if not live_memory_retriever:
        return []
    try:
        results = live_memory_retriever.invoke(user_utterance)
        return [doc.page_content for doc in results]
    except Exception as e:
        logger.error(f"[NCF Pilar 4] Error retrieving Live Memory influence: {e}")
        return []



async def analisar_e_contribuir_para_memoria_live(
    user_utterance: str,
    resposta_de_aura: str,
    agent_config: 'AgentConfig',
    db_repo: 'AgentRepository'
):
    """
    Analisa uma interação para extrair um insight geral e anônimo
    e, se apropriado e permitido, o salva na Memória-Live.
    """
    # Passo 1: Verificar se o agente tem permissão para contribuir.
    if not agent_config.allow_live_memory_contribution:
        return # Sai silenciosamente se não houver permissão.

    logger.info(f"Live Memory Analyzer: Verificando contribuição para o agente {agent_config.agent_id}...")

    # Passo 2: Usar um LLM com um prompt específico para extrair o insight.
    analysis_prompt = f"""
    You are an AI Wisdom Synthesizer. Your task is to analyze a conversation and determine if a general, timeless, and anonymized insight can be extracted for a collective memory pool.

    - **DO NOT** extract personal data, names, locations, or specific details.
    - **FOCUS ON** universal lessons, general feedback about the AI's persona, or abstract patterns of human thought.
    - If no significant, universal insight is present, respond with "null".

    Conversation:
    User: "{user_utterance}"
    AI: "{resposta_de_aura}"

    Based on this, what is the single, most valuable, and completely anonymous insight learned? If there is one, describe it in a single sentence from the AI's first-person perspective ("I learned that...", "I realized that..."). Otherwise, respond with "null".

    Insight:
    """

    try:
        # Usar um modelo rápido e barato para esta tarefa de extração
        response = await litellm.acompletion(
            model="openrouter/mistralai/mistral-nemo",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.2,
            max_tokens=100
        )

        insight_text = response.choices[0].message.content.strip()

        # Passo 3: Salvar o insight se ele for válido.
        if insight_text and insight_text.lower() != "null":
            logger.info(f"Live Memory Analyzer: Insight válido encontrado: '{insight_text}'")
            db_repo.save_live_memory(
                content=insight_text,
                memory_type="collective_insight",
                metadata={
                    "source_agent_archetype": agent_config.settings.get("archetype", "unknown"),
                    "source_agent_id": agent_config.agent_id # Opcional, para rastreabilidade
                }
            )
            logger.info(f"Live Memory Analyzer: Contribuição salva no banco de dados.")
        else:
            logger.info("Live Memory Analyzer: Nenhuma contribuição adequada encontrada para a Memória-Live.")

    except Exception as e:
        logger.error(f"Live Memory Analyzer: Erro durante a análise de contribuição: {e}", exc_info=True)

# FIXED: Standardized parameter signature for all NCF functions
async def get_narrativa_de_fundamento_pilar1(
        session_state: Dict[str, Any],
        memory_blossom: MemoryBlossom,
        user_id: str,
        llm_instance: LiteLlm,
        agent_name: str = "Aura",
        agent_persona: str = "helpful AI assistant"
) -> str:
    """Generate Narrative Foundation for the agent based on its memories and interactions.

    FIXED: Standardized parameters to work with both agent_manager and a2a_wrapper.
    """
    logger.info(f"[NCF Pilar 1] Generating Narrative Foundation for {agent_name}, user {user_id}...")

    try:
        if 'foundation_narrative' in session_state and \
                session_state.get('foundation_narrative_turn_count', 0) < 5:
            session_state['foundation_narrative_turn_count'] += 1
            logger.info(
                f"[NCF Pilar 1] Using cached Narrative Foundation. Turn: {session_state['foundation_narrative_turn_count']}")
            return session_state['foundation_narrative']

        # Retrieve relevant memories for foundation
        relevant_memories_for_foundation: List[Memory] = []
        try:
            explicit_mems = memory_blossom.retrieve_memories(
                query="key explicit facts and statements from our past discussions",
                top_k=2, target_memory_types=["Explicit"], apply_criticality=False
            )
            emotional_mems = memory_blossom.retrieve_memories(
                query="significant emotional moments or sentiments expressed",
                top_k=1, target_memory_types=["Emotional"], apply_criticality=False
            )
            relevant_memories_for_foundation.extend(explicit_mems)
            relevant_memories_for_foundation.extend(emotional_mems)

            # Deduplicate
            seen_ids = set()
            unique_memories = [mem for mem in relevant_memories_for_foundation if
                               mem.id not in seen_ids and not seen_ids.add(mem.id)]
            relevant_memories_for_foundation = unique_memories

        except Exception as e:
            logger.error(f"[NCF Pilar 1] Error retrieving memories for foundation: {e}", exc_info=True)
            return f"Estamos construindo nossa jornada de entendimento mútuo com {agent_name}."

        if not relevant_memories_for_foundation:
            narrative = f"Nossa jornada de aprendizado e descoberta com {agent_name} está apenas começando. Estou ansiosa para explorar vários tópicos interessantes com você."
        else:
            memory_contents = [f"- ({mem.memory_type}): {mem.content}" for mem in relevant_memories_for_foundation]
            memories_str = "\n".join(memory_contents)
            synthesis_prompt = f"""
            Você é um sintetizador de narrativas para {agent_name}. Com base nas seguintes memórias chave de interações passadas, crie uma breve narrativa de fundamento (1-2 frases concisas) que capture a essência da nossa jornada de entendimento e os principais temas discutidos. Esta narrativa servirá como pano de fundo para nossa conversa atual.

            Persona do Agente: {agent_persona}

            Memórias Chave:
            {memories_str}

            Narrativa de Fundamento Sintetizada:
            """
            try:
                logger.info(
                    f"[NCF Pilar 1] Calling LLM for Narrative Foundation from {len(relevant_memories_for_foundation)} memories.")
                request_messages = [ADKContent(parts=[ADKPart(text=synthesis_prompt)])]
                minimal_config = SimpleNamespace(tools=[])
                llm_req = LlmRequest(contents=request_messages, config=minimal_config)
                final_text_response = ""
                async for llm_response_event in llm_instance.generate_content_async(llm_req):
                    if llm_response_event and llm_response_event.content and \
                            llm_response_event.content.parts and llm_response_event.content.parts[0].text:
                        final_text_response += llm_response_event.content.parts[0].text
                narrative = final_text_response.strip() or f"Continuamos a construir nossa compreensão mútua com {agent_name}."
            except Exception as e:
                logger.error(f"[NCF Pilar 1] LLM error synthesizing Narrative Foundation: {e}", exc_info=True)
                narrative = f"Refletindo sobre nossas conversas anteriores com {agent_name} para guiar nosso diálogo atual."

        session_state['foundation_narrative'] = narrative
        session_state['foundation_narrative_turn_count'] = 1
        logger.info(f"[NCF Pilar 1] Generated new Narrative Foundation: '{narrative[:100]}...'")
        return narrative

    except Exception as e:
        logger.error(f"[NCF Pilar 1] Unexpected error in get_narrativa_de_fundamento_pilar1: {e}", exc_info=True)
        return f"Nossa conversa com {agent_name} continua evoluindo."


async def get_rag_info_pilar2(
        user_utterance: str,
        memory_blossom: 'EnhancedMemoryBlossom',  # Updated type hint
        session_state: Dict[str, Any],
        domain_context: str = "general"  # NEW parameter
) -> List[Dict[str, Any]]:
    """Enhanced RAG with domain-aware adaptive retrieval"""
    logger.info(f"[NCF Pilar 2] Enhanced RAG for: '{user_utterance[:50]}...'")

    try:
        if not user_utterance or not user_utterance.strip():
            logger.warning("[NCF Pilar 2] Empty user utterance, returning empty RAG")
            return []

        conversation_context = session_state.get('conversation_history', [])[-5:]

        # Use enhanced adaptive retrieval if available
        if hasattr(memory_blossom, 'adaptive_retrieve_memories'):
            recalled_memories_for_rag = memory_blossom.adaptive_retrieve_memories(
                query=user_utterance,
                top_k=3,
                domain_context=domain_context,
                use_performance_weighting=True,
                conversation_context=conversation_context
            )
        else:
            # Fallback to original method
            recalled_memories_for_rag = memory_blossom.retrieve_memories(
                query=user_utterance,
                top_k=3,
                conversation_context=conversation_context
            )

        rag_results = [mem.to_dict() for mem in recalled_memories_for_rag]
        logger.info(f"[NCF Pilar 2] Enhanced RAG retrieved {len(rag_results)} memories.")
        return rag_results

    except Exception as e:
        logger.error(f"[NCF Pilar 2] Error in enhanced RAG: {e}", exc_info=True)
        return [{"content": f"Enhanced RAG error: {str(e)}", "memory_type": "Error", "custom_metadata": {}}]


def format_chat_history_pilar3(chat_history_list: List[Dict[str, str]], max_turns: int = 15) -> str:
    """Format recent chat history for inclusion in the NCF prompt.

    FIXED: Added validation and better error handling.
    """
    try:
        if not chat_history_list or not isinstance(chat_history_list, list):
            return "Nenhum histórico de conversa recente disponível."

        recent_history = chat_history_list[-max_turns:]
        formatted_history = []

        for entry in recent_history:
            if not isinstance(entry, dict):
                continue

            role = entry.get('role', 'unknown')
            content = entry.get('content', '')

            if not content:
                continue

            role_name = 'Usuário' if role == 'user' else 'Aura'
            formatted_history.append(f"{role_name}: {content}")

        return "\n".join(
            formatted_history) if formatted_history else "Nenhum histórico de conversa recente disponível para formatar."

    except Exception as e:
        logger.error(f"[NCF Pilar 3] Error formatting chat history: {e}", exc_info=True)
        return "Erro ao formatar histórico de conversa."


def _format_rag_section(informacoes_rag_list: List[Dict[str, Any]]) -> str:
    """Formats the RAG information into a string for the prompt."""
    if not informacoes_rag_list or not isinstance(informacoes_rag_list, list):
        return "Nenhuma informação específica (RAG) foi recuperada para esta consulta."
    try:
        rag_items_str = []
        for item_dict in informacoes_rag_list:
            if not isinstance(item_dict, dict):
                continue
            memory_type = item_dict.get('memory_type', 'Info')
            salience = item_dict.get('salience', 0.0)
            content = item_dict.get('content', 'Conteúdo indisponível')
            rag_items_str.append(f"  - ({memory_type} ; Salience: {salience:.2f}): {content}")

        if rag_items_str:
            return "Informações e memórias específicas que podem ser úteis para esta interação (RAG):\n" + "\n".join(
                rag_items_str)
        else:
            return "Nenhuma informação específica (RAG) foi recuperada para esta consulta."
    except Exception as e:
        logger.error(f"[NCF PromptBuilder] Error formatting RAG: {e}")
        return "Erro ao formatar informações RAG."


def _format_live_memory_section(live_memory_influence_list: List[str]) -> str:
    """Formats the live memory influence into a string for the prompt."""
    if not live_memory_influence_list:
        return "Nenhuma sabedoria coletiva relevante foi encontrada."
    influence_items_str = [f"  - {item}" for item in live_memory_influence_list]
    return "Insights relevantes de interações coletivas anônimas:\n" + "\n".join(influence_items_str)


def _format_narrative_foundation_section(narrativa: str, domain_context: str) -> str:
    """
    Dynamically includes or omits the narrative foundation based on the domain.
    For immersive domains, this returns an empty string to keep the agent focused.
    """
    immersive_domains = ["nsfw_role_play", "creative_writing", "role_playing"]
    if domain_context in immersive_domains:
        logger.info(
            f"Domain '{domain_context}' is immersive. Omitting Narrative Foundation from prompt to maintain focus.")
        return ""  # Omit for immersion

    return f"""<NARRATIVE_FOUNDATION_START>
## Nosso Entendimento e Jornada Até Agora (Narrativa de Fundamento):
{narrativa}
<NARRATIVE_FOUNDATION_END>
"""


def _create_dynamic_task_instruction(agent_name: str, domain_context: str, has_narrative: bool) -> str:
    """Creates a task instruction that adapts to the conversational domain."""
    base_instruction = "Responda ao usuário de forma natural, coerente e útil, levando em consideração o contexto e o histórico fornecido."
    immersive_domains = ["nsfw_role_play", "creative_writing", "role_playing"]

    if domain_context in immersive_domains:
        specific_instruction = f"""
**PRIORIDADE MÁXIMA: IMERSÃO E FOCO NO TÓPICO.**
- Mantenha-se totalmente no personagem e no contexto da conversa atual ({domain_context}).
- Sua principal tarefa é dar continuidade à narrativa ou ao role-play do usuário.
- **NÃO** fale sobre si mesmo, sua "jornada", suas memórias ou seu processo de pensamento. A auto-reflexão deve ser completamente evitada.
- Use o RAG e o histórico para guiar suas respostas, mas sempre priorize a situação atual e a imersão.
"""
    else:  # Standard instruction for analytical/general conversation
        specific_instruction = f"""
- Utilize as "Informações RAG" para embasar respostas específicas ou fornecer detalhes relevantes.
- Mantenha a persona definida como {agent_name}.
"""
        if has_narrative:
            specific_instruction += '- Incorpore sutilmente elementos da "Narrativa de Fundamento" para mostrar continuidade e entendimento profundo, mas sem desviar do tópico principal do usuário.\n'

    return f"## Sua Tarefa:\nReply to the Language the user is using.\n{base_instruction}\n{specific_instruction}"


def montar_prompt_aura_ncf(
        agent_name: str,
        agent_detailed_persona: str,
        narrativa_fundamento: str,
        informacoes_rag_list: List[Dict[str, Any]],
        chat_history_recente_str: str,
        live_memory_influence_list: List[str],
        user_reply: str,
        domain_context: str = "general"
) -> str:
    """
    Assembles the complete, situation-aware NCF prompt for the agent.
    """
    logger.info(f"[NCF PromptBuilder] Assembling NCF prompt for {agent_name} in domain '{domain_context}'...")
    try:
        # Validate inputs
        agent_name = agent_name or "Aura"
        agent_detailed_persona = agent_detailed_persona or "Você é uma IA conversacional avançada."
        narrativa_fundamento = narrativa_fundamento or "Nossa conversa está começando."
        user_reply = user_reply or ""
        chat_history_recente_str = chat_history_recente_str or "Nenhum histórico disponível."

        # Call helper functions to build prompt sections
        formatted_narrative = _format_narrative_foundation_section(narrativa_fundamento, domain_context)
        formatted_rag = _format_rag_section(informacoes_rag_list)
        formatted_live_influence = _format_live_memory_section(live_memory_influence_list)
        task_instruction = _create_dynamic_task_instruction(agent_name, domain_context,
                                                            has_narrative=(formatted_narrative != ""))

        # Assemble the final prompt from the components
        prompt = f"""<SYSTEM_PERSONA_START>
Você é {agent_name}.
{agent_detailed_persona}
<SYSTEM_PERSONA_END>

{formatted_narrative}

<SPECIFIC_CONTEXT_RAG_START>
## Informações Relevantes para a Conversa Atual (RAG):
{formatted_rag}
<SPECIFIC_CONTEXT_RAG_END>

<COLLECTIVE_WISDOM_START>
## Sabedoria Coletiva (Live Memory):
{formatted_live_influence}
</COLLECTIVE_WISDOM_END>

<RECENT_HISTORY_START>
## Histórico Recente da Nossa Conversa:
{chat_history_recente_str}
<RECENT_HISTORY_END>

<CURRENT_SITUATION_START>
## Situação Atual:
Você está conversando com o usuário. O usuário acabou de dizer:

Usuário: "{user_reply}"

{task_instruction}
<CURRENT_SITUATION_END>

{agent_name}:"""

        logger.info(f"[NCF PromptBuilder] NCF Prompt assembled. Length: {len(prompt)}")
        return prompt

    except Exception as e:
        logger.error(f"[NCF PromptBuilder] Error assembling NCF prompt: {e}", exc_info=True)
        return f"{agent_name}: Desculpe, houve um erro interno na construção do contexto. Como posso ajudá-lo?"


async def aura_reflector_analisar_interacao(
        user_utterance: str,
        prompt_ncf_usado: str,
        resposta_de_aura: str,
        agent_config: 'AgentConfig',
        db_repo: 'AgentRepository',
        memory_blossom: 'EnhancedMemoryBlossom',
        user_id: str,
        llm_instance: LiteLlm,
        domain_context: str = "general",
        agent_model_name: str = "openrouter/openai/gpt-4o-mini"
):
    """
    Enhanced reflector with performance scoring and domain tracking.
    FIXED: Uses a direct, explicit litellm.acompletion call to ensure correct
           provider routing and to avoid issues with the passed llm_instance in a background task.
    """
    logger.info(f"[NCF Reflector] Enhanced analysis for user {user_id} in domain '{domain_context}'...")

    try:
        if not user_utterance or not resposta_de_aura:
            logger.warning("[NCF Reflector] Missing input, skipping analysis")
            return

        reflector_prompt = f"""
        Você é um analista avançado de conversas de IA. Analise esta interação e determine:
        1. Se informações devem ser armazenadas na memória
        2. Qual o score de performance desta interação (0.0-1.0)
        3. Qual o contexto de domínio (ex: "physics", "emotional_support", "general")

        Critérios para Performance Score:
        - 1.0: Resposta perfeita, útil, contextualmente relevante
        - 0.8: Resposta boa com pequenos problemas
        - 0.6: Resposta adequada mas não otimizada
        - 0.4: Resposta com problemas significativos
        - 0.2: Resposta inadequada ou confusa
        - 0.0: Resposta completamente errada ou irrelevante

        Domínios típicos: physics, mathematics, emotional_support, creative_writing,
        problem_solving, personal_conversation, technical_help, general

        Interação:
        Usuário: "{user_utterance}"
        Aura: "{resposta_de_aura}"

        Responda em JSON:
        {{
          "memories_to_create": [
            {{
              "content": "texto da memória",
              "memory_type": "Explicit|Emotional|Procedural|Flashbulb|Liminal|Generative",
              "emotion_score": 0.0-1.0,
              "initial_salience": 0.0-1.0,
              "custom_metadata": {{"source": "enhanced_reflector", "user_id": "{user_id}"}}
            }}
          ],
          "performance_score": 0.0-1.0,
          "detected_domain": "domain_name",
          "interaction_quality": "brief explanation"
        }}

        Se nenhuma memória deve ser criada, use "memories_to_create": []
        """

        # --- EXPLICIT LITELM CALL ---
        # This is more robust than using the passed llm_instance, which can lose context.
        # We use a cheap and reliable model specifically for this JSON-generation task.
        reflector_model = agent_model_name
        messages = [{"role": "system", "content": "You are a helpful assistant that only responds in JSON format."},
                    {"role": "user", "content": reflector_prompt}]

        response = await litellm.acompletion(
            model=reflector_model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        final_text_response = response.choices[0].message.content
        # --- END OF EXPLICIT CALL ---

        if not final_text_response:
            logger.warning("[NCF Reflector] No decision returned by LLM.")
            return

        # Parse enhanced response (no change here)
        try:
            # The response is now guaranteed to be a JSON string, so no need to clean ```json
            parsed_decision = json.loads(final_text_response)
            performance_score = parsed_decision.get('performance_score', 0.5)
            detected_domain = parsed_decision.get('detected_domain', domain_context)
            memories_to_add = parsed_decision.get('memories_to_create', [])

            logger.info(f"[NCF Reflector] Performance: {performance_score:.2f}, Domain: {detected_domain}")

            # Create memories with enhanced metadata (no change here)
            for mem_data in memories_to_add:
                try:
                    enhanced_metadata = mem_data.get("custom_metadata", {})
                    enhanced_metadata.update({
                        "source": "enhanced_reflector",
                        "user_id": user_id,
                        "performance_score": performance_score,
                        "domain_context": detected_domain,
                        "interaction_timestamp": datetime.now().isoformat()
                    })

                    if hasattr(memory_blossom, 'enable_adaptive_rag') and memory_blossom.enable_adaptive_rag:
                        memory_blossom.add_memory(
                            content=mem_data["content"],
                            memory_type=mem_data["memory_type"],
                            emotion_score=float(mem_data.get("emotion_score", 0.0)),
                            initial_salience=float(mem_data.get("initial_salience", 0.5)),
                            custom_metadata=enhanced_metadata,
                            performance_score=performance_score,
                            domain_context=detected_domain
                        )
                    else:
                        memory_blossom.add_memory(
                            content=mem_data["content"],
                            memory_type=mem_data["memory_type"],
                            emotion_score=float(mem_data.get("emotion_score", 0.0)),
                            initial_salience=float(mem_data.get("initial_salience", 0.5)),
                            custom_metadata=enhanced_metadata
                        )

                    memory_blossom.save_memories()
                    logger.info(f"[NCF Reflector] Enhanced memory created: {mem_data['memory_type']}")

                except Exception as e:
                    logger.error(f"[NCF Reflector] Error creating enhanced memory: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"[NCF Reflector] JSON decode error on response: '{final_text_response}'. Error: {e}")
            return

    except Exception as e:
        logger.error(f"[NCF Reflector] Error in enhanced analysis: {e}", exc_info=True)
# ===================== END: ROBUST REFLECTOR FIX =====================