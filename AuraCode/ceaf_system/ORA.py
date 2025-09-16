import asyncio
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
import random
import numpy as np
from collections import deque

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from litellm import acompletion

# Import our custom modules
from .AMA import MemoryExperience
from .MCL import MetacognitiveControlLoop, CoherenceMetrics
from .logprob_analyzer import calculate_token_confidence_from_top_logprobs
from .NCIM import narrative_embedding_model

# VRE will be passed in, so no direct import needed here, but it's good practice
# from VRE import VirtueReasoningEngine

logger = logging.getLogger(__name__)


# State definition for LangGraph
class AgentState(TypedDict):
    """State passed between agents in the graph"""
    messages: List[BaseMessage]
    current_query: str
    memory_context: List[Dict[str, Any]]
    coherence_metrics: Optional[Dict[str, float]]
    active_failures: List[str]
    narrative_context: Optional[str]
    virtue_considerations: List[str]
    loss_insights: List[Dict[str, Any]]
    live_memory_influence: Optional[List[str]]  # Add the new state field
    response_draft: Optional[str]
    metadata: Dict[str, Any]
    feedback_classification: Optional[str]
    memory_context_salience: float
    loss_insights_salience: float
    virtue_considerations_salience: float


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    name: str
    model: str = "openrouter/openai/gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: str = ""


def extract_answer(text: str) -> str:
    # This is a simplified version from the DeepConf helper.py
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if ans and ans[0] == "{":
            # Handle nested braces
            stack = 1; a = "";
            for c in ans[1:]:
                if c == "{": stack += 1
                elif c == "}": stack -= 1
                if stack == 0: break
                a += c
            return a.strip()
        else:
            return ans.split("$")[0].strip()
    return "" # Return empty string if no answer found

def weighted_majority_vote(answers: list, weights: list) -> str:
    if not answers: return None
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None and answer != "":
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    if not answer_weights: return None
    return max(answer_weights, key=answer_weights.get)


class CEAFOrchestrator:
    def __init__(self, openrouter_api_key: str, memory_architecture: Any, mcl: MetacognitiveControlLoop, lcam: Any,
                 ncim: Any, vre: Any):
        self.openrouter_api_key = openrouter_api_key
        self.memory = memory_architecture
        self.mcl = mcl
        self.lcam = lcam
        self.ncim = ncim
        self.vre = vre
        self.agents = self._initialize_agents()
        self.checkpointer = InMemorySaver()
        self.workflow = self._build_workflow()
        logger.info("Initialized CEAF Orchestrator with LCAM, NCIM, and VRE integration")

    def _initialize_agents(self) -> Dict[str, AgentConfig]:
        return {
            "ora": AgentConfig(name="Orchestrator/Responder Agent",
                               system_prompt="You are the Orchestrator/Responder Agent (ORA)..."),
            "memory_analyst": AgentConfig(name="Memory Pattern Analyst", temperature=0.5,
                                          system_prompt="You analyze memory patterns..."),
            "narrative_weaver": AgentConfig(name="Narrative Coherence Weaver", temperature=0.8,
                                            system_prompt="You weave coherent narratives..."),
            "virtue_engineer": AgentConfig(name="Virtue & Reasoning Engineer", temperature=0.6,
                                           system_prompt="You ensure principled reasoning..."),
            "loss_cataloger": AgentConfig(name="Loss Pattern Cataloger", temperature=0.4,
                                          system_prompt="You analyze and catalog failure patterns..."),
            "edge_navigator": AgentConfig(name="Edge of Coherence Navigator", temperature=0.9,
                                          system_prompt="You help navigate the edge..."),
            "feedback_classifier": AgentConfig(name="Feedback Classifier", temperature=0.0, max_tokens=10,
                                               system_prompt="You are a text classification agent. Respond with ONLY ONE of the following words: 'positive', 'neutral', or 'negative_critique'."),


            "fact_extractor": AgentConfig(name="Factual Extractor", temperature=0.0,
                                          system_prompt="You are a highly precise information extraction agent. Analyze the provided text and extract concrete facts as a JSON list of strings. If there are no facts, return an empty JSON list []."),


            "narrative_synthesizer": AgentConfig(
                name="Narrative Synthesizer",
                temperature=0.3,
                max_tokens=500,
                system_prompt="You are a master synthesizer. Your job is to take a set of disparate data points about an AI's internal state and weave them into a single, coherent, causal, first-person narrative. The output must be an irreducible story that explains the 'why' behind the AI's current perspective. Do not list the inputs; integrate them."
            )
        }


    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve_memory", self._retrieve_memory_context)
        workflow.add_node("assess_coherence", self._assess_coherence)
        workflow.add_node("get_loss_insights", self._get_loss_insights)
        workflow.add_node("get_virtue_input", self._get_virtue_input)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("learn_from_interaction", self._learn_from_interaction)
        workflow.add_node("update_identity", self._update_identity)

        workflow.set_entry_point("retrieve_memory")
        workflow.add_edge("retrieve_memory", "assess_coherence")
        workflow.add_edge("assess_coherence", "get_loss_insights")
        workflow.add_edge("get_loss_insights", "get_virtue_input")
        workflow.add_edge("get_virtue_input", "generate_response")
        workflow.add_edge("generate_response", "learn_from_interaction")
        workflow.add_edge("learn_from_interaction", "update_identity")
        workflow.add_edge("update_identity", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def _streaming_call_with_deepconf(self, messages: list, mcl_params: dict, state: AgentState,
                                            early_stop_threshold: float = None) -> dict:
        """
        Performs a single, streaming LLM call and applies client-side DeepConf logic,
        now using the session-specific model override if present.
        """
        full_response_text = ""
        all_token_confs = []
        is_early_stopped = False

        # --- START MODIFICATION: Determine which model to use for this specific call ---
        model_to_use = self.agents["ora"].model # Start with the default for ORA
        session_overrides = state.get("session_overrides")
        if session_overrides and 'model' in session_overrides:
            model_to_use = session_overrides['model']
        # --- END MODIFICATION ---

        WINDOW_SIZE = mcl_params.get("confidence_window", 2048)
        token_conf_window = deque(maxlen=WINDOW_SIZE)

        stream = await acompletion(
            model=model_to_use,  # <-- USE THE OVERRIDE-AWARE VARIABLE
            messages=messages,
            temperature=mcl_params.get("temperature", 0.7),
            max_tokens=self.agents["ora"].max_tokens,
            stream=True,
            logprobs=True,
            top_logprobs=20
        )

        async for chunk in stream:
            choice = chunk.choices[0]
            if choice.delta.content:
                full_response_text += choice.delta.content

            if choice.logprobs and choice.logprobs.content:
                new_conf = calculate_token_confidence_from_top_logprobs(choice.logprobs.content[0])
                all_token_confs.append(new_conf)
                token_conf_window.append(new_conf)

                if early_stop_threshold and len(token_conf_window) >= WINDOW_SIZE:
                    avg_conf = sum(token_conf_window) / len(token_conf_window)
                    if avg_conf < early_stop_threshold:
                        is_early_stopped = True
                        break  # Stop processing the stream

            if choice.finish_reason:
                break

        sliding_means = [np.mean(all_token_confs[i:i + WINDOW_SIZE]) for i in
                         range(len(all_token_confs) - WINDOW_SIZE + 1)]
        min_conf_for_trace = min(sliding_means) if sliding_means else (
            np.mean(all_token_confs) if all_token_confs else 20.0)

        return {
            "text": full_response_text,
            "min_conf": min_conf_for_trace,
            "token_count": len(all_token_confs),
            "stopped_early": is_early_stopped,
            "model_used": model_to_use # Add for debugging
        }

    async def _call_agent(self, agent_name: str, prompt: str, state: AgentState) -> str:
        agent_config = self.agents[agent_name]

        # --- START MODIFICATION: Use session override model if available ---
        model_to_use = agent_config.model
        session_overrides = state.get("session_overrides")
        if session_overrides and 'model' in session_overrides:
            model_to_use = session_overrides['model']
            logger.info(f"ORA Specialist '{agent_name}' using override model: {model_to_use}")
        # --- END MODIFICATION ---

        messages = [{"role": "system", "content": agent_config.system_prompt}, {"role": "user", "content": prompt}]
        try:
            response = await acompletion(
                model=model_to_use,  # <-- USE THE OVERRIDDEN MODEL
                messages=messages,
                temperature=agent_config.temperature,
                max_tokens=agent_config.max_tokens,
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            # Make sure to handle potential None or empty choices
            if response.choices and response.choices[0].message.content:
                 return response.choices[0].message.content.strip()
            return f"Error: Agent {agent_name} returned an empty response."
        except Exception as e:
            logger.error(f"Error calling agent {agent_name}: {e}", exc_info=True)
            if self.mcl: self.mcl.register_failure(f"agent_call_failure_{agent_name}", {"error": str(e)})
            return f"Error: Could not get a response from the {agent_name} agent."

    async def _retrieve_memory_context(self, state: AgentState) -> AgentState:
        if not self.memory:
            state["memory_context"] = []
            return state
        try:
            memories, salience = self.memory.retrieve_with_loss_context(
                query=state["current_query"], k=5, include_failures=True, failure_weight=0.4
            )
            memory_context = []
            for mem in memories:
                mem_dict = asdict(mem)
                mem_dict.pop('embedding', None)
                memory_context.append(mem_dict)
            state["memory_context"] = memory_context
            state["memory_context_salience"] = salience
        except Exception as e:
            logger.error(f"Error in _retrieve_memory_context: {e}")
            state["memory_context"] = []
        return state

    async def _assess_coherence(self, state: AgentState) -> AgentState:
        if not self.mcl:
            state["coherence_metrics"] = {"overall_coherence": 0.7}
            return state

        deepconf_metadata = state.get("metadata", {})


        # Retrieve the list of all valid reasoning path texts from the state.
        valid_trace_texts = deepconf_metadata.get("valid_trace_texts", [])

        narrative_score = 0.5  # Default neutral score

        # If there were valid traces, calculate the score based on ALL of them.
        if valid_trace_texts and self.ncim:
            logger.info(f"ORA: Assessing narrative coherence across {len(valid_trace_texts)} valid thought paths...")
            narrative_scores_array = self.ncim.calculate_narrative_coherence_scores(
                potential_responses=valid_trace_texts
            )
            narrative_score = np.mean(narrative_scores_array)
        elif state.get("response_draft") and self.ncim:
            # Fallback to checking only the final answer if the list isn't available for some reason
            logger.warning(
                "ORA: Valid trace texts not found in state, falling back to final answer for narrative score.")
            narrative_score = self.ncim.calculate_narrative_coherence_score(
                potential_responses=[state.get("response_draft")]
            )


        # Call the MCL's assessment method with the new, more robust narrative_score
        metrics = await self.mcl.assess_coherence_from_deepconf(
            deepconf_metadata=deepconf_metadata,
            narrative_score=narrative_score
        )

        state["coherence_metrics"] = asdict(metrics)
        return state

    async def _get_loss_insights(self, state: AgentState) -> AgentState:
        if not self.lcam:
            state["loss_insights"] = []
            return state
        try:
            insights, salience = self.lcam.get_insights_for_context(
                query=state["current_query"],
                memory_context=state.get("memory_context", [])
            )
            state["loss_insights"] = insights
            state["loss_insights_salience"] = salience
        except Exception as e:
            logger.error(f"Error in _get_loss_insights: {e}")
            state["loss_insights"] = []

        return state

    async def _get_virtue_input(self, state: AgentState) -> AgentState:
        if not self.vre:
            state["virtue_considerations"] = []
            return state
        try:
            considerations, salience = self.vre.get_virtue_considerations(state)
            state["virtue_considerations"] = considerations
            state["virtue_considerations_salience"] = salience
        except Exception as e:
            logger.error(f"Error in _get_virtue_input: {e}")
            state["virtue_considerations"] = ["Error retrieving virtue considerations."]
        return state

    async def _classify_user_feedback(self, user_message: str) -> str:
        """Uses an LLM to classify the user's feedback on the previous turn."""
        prompt = f"""Analyze the following user feedback and classify it.

        User feedback: "{user_message}"

        Possible classifications:
        - 'positive': The user is expressing satisfaction, agreement, or praise.
        - 'neutral': The user is asking a follow-up question, changing the subject, or providing neutral information.
        - 'negative_critique': The user is expressing dissatisfaction, pointing out a flaw, correcting the AI, or calling the response generic, abstract, or a failure.

        What is the classification of the feedback?
        """
        classification = await self._call_agent("feedback_classifier", prompt, {})

        # Ensure the output is one of the valid classifications
        if classification in ['positive', 'neutral', 'negative_critique']:
            logger.info(f"User feedback classified as: {classification}")
            return classification
        else:
            logger.warning(
                f"Feedback classifier returned unexpected value: '{classification}'. Defaulting to 'neutral'.")
            return 'neutral'

    async def _learn_from_interaction(self, state: AgentState) -> AgentState:
        if not self.memory or not self.mcl:
            return state

        # --- Part 1: Standard Coherence and Success/Failure Analysis (Unchanged) ---
        last_human_message = state.get("current_query", "")
        feedback_classification = await self._classify_user_feedback(last_human_message)
        state["feedback_classification"] = feedback_classification
        coherence = state.get("coherence_metrics", {}).get("overall_coherence", 0.5)

        is_critique = (feedback_classification == 'negative_critique')
        success = not is_critique and coherence > 0.55

        failure_pattern = None
        if not success:
            failure_pattern = "user_critique_of_response" if is_critique else 'coherence_loss'

        # --- Part 2: Create Thematic Memory (The Agent's "Experience") ---
        # This memory captures the agent's subjective experience of the interaction.
        thematic_content = f"I was asked about '{state['current_query']}' and I responded '{state.get('response_draft', '')[:200]}...'. I reflect that this interaction was a {'success' if success else 'challenge'}."

        thematic_experience = MemoryExperience(
            content=thematic_content,
            timestamp=datetime.now(),
            experience_type='success' if success else 'failure',
            context={"coherence": state.get("coherence_metrics", {}), "feedback_class": feedback_classification},
            outcome_value=1.0 if success else -0.5,
            learning_value=abs(coherence - 0.5) + (0.4 if not success else 0.2),
            failure_pattern=failure_pattern,
            metadata={"interaction_id": state.get("metadata", {}).get("interaction_id"), "memory_style": "thematic"}
        )
        self.memory.add_experience(thematic_experience)

        if not success and self.lcam:
            self.lcam.catalog_failure(thematic_experience)

        # --- Part 3: NEW - Extract and Store Concrete Factual Memories ---
        # This process runs in parallel to ensure the agent doesn't lose important details.

        fact_extractor_prompt = f"""
        Analyze the following dialogue turn and extract any concrete, specific, and verifiable facts.
        Facts can include names, dates, specific numbers, locations, relationships, or declared actions.
        Do NOT extract opinions, summaries, or abstract concepts.
        If no concrete facts are present, you MUST respond with an empty JSON list: [].

        Dialogue:
        - "{state['current_query']}"
        - "{state.get('response_draft', '')}"

        Format the output as a valid JSON list of strings.
        Example: ["Caroline is a transgender woman.", "Melanie ran a charity race on the sunday before 25 May 2023.", "Melanie has 3 children."]
        """

        try:
            # Use the specialized 'fact_extractor' agent with a cheap, fast model.
            extracted_facts_str = await self._call_agent("fact_extractor", fact_extractor_prompt, state)

            # The response might be inside a code block, so we clean it up
            if "```json" in extracted_facts_str:
                extracted_facts_str = extracted_facts_str.split("```json\n")[1].split("```")[0]

            fact_list = json.loads(extracted_facts_str)

            if isinstance(fact_list, list):
                for fact in fact_list:
                    # Create a new, high-salience factual memory
                    factual_experience = MemoryExperience(
                        content=fact,
                        timestamp=datetime.now(),
                        experience_type='explicit_fact',  # A new, specific type for easy retrieval
                        context={"source": "fact_extractor"},
                        outcome_value=1.0,  # Storing a fact is always a success
                        learning_value=0.9,  # Facts are highly valuable for recall
                        metadata={"interaction_id": state.get("metadata", {}).get("interaction_id"),
                                  "memory_style": "factual"}
                    )
                    self.memory.add_experience(factual_experience)
                    logger.info(f"CEAF: Extracted and stored factual memory: '{fact}'")
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                f"CEAF: Fact Extractor did not return a valid JSON list. Response: '{extracted_facts_str}'. Error: {e}")
        except Exception as e:
            logger.error(f"CEAF: An unexpected error occurred in the fact extraction process: {e}", exc_info=True)

        return state

    async def _update_identity(self, state: AgentState) -> AgentState:
        if not self.ncim:
            return state
        try:
            query = state["current_query"]
            response = state["response_draft"]

            last_human_message = state["current_query"]
            feedback_class = state.get("feedback_classification", "neutral")
            coherence = state.get("coherence_metrics", {}).get("overall_coherence", 0.5)
            success = (feedback_class != 'negative_critique') and coherence > 0.55

            loss_insights = state.get("loss_insights", [])
            outcome = "a success" if success else "a challenge (failure)"
            lesson = "This reinforced my existing knowledge."
            if not success:
                lesson = "This was a difficult interaction. "
                if loss_insights:
                    lesson += f"It reminded me of past failures related to '{loss_insights[0]['failure_pattern']}'. I must learn to be more careful in such situations."
                else:
                    lesson += "It highlighted a new area where my understanding is weak. I need to reflect on this."
            interaction_summary = (
                f"I was asked: '{query}'. "
                f"I responded: '{response[:200]}...'. "
                f"The outcome was considered {outcome}. "
                f"The key lesson for me is: {lesson}"
            )
            update_prompt = self.ncim.get_update_prompt(interaction_summary)
            new_narrative = await self._call_agent("narrative_weaver", update_prompt, state)
            self.ncim.update_identity(new_narrative)
        except Exception as e:
            logger.error(f"Error during identity update: {e}", exc_info=True)
        return state

    async def _generate_response(self, state: AgentState) -> AgentState:
        """
        Generates a response by first synthesizing a causally dense internal context
        and then using that context to inform the final generation process (DeepConf or Fast Path).
        """
        precision_mode = state.get("metadata", {}).get("precision_mode", False)
        logger.info(f"ORA: Entering generation node. Precision Mode: {precision_mode}")

        # Get the dynamically adjusted parameters from the MCL
        mcl_params = self.mcl.get_next_turn_parameters(precision_mode=precision_mode)

        # --- PHASE 1: GATHER ALL RAW CONTEXT FOR SYNTHESIS ---
        identity_narrative = self.ncim.get_current_identity()
        metacognitive_summary = self.mcl.get_metacognitive_summary()

        # Format RAG memories for context
        memory_context_list = state.get("memory_context", [])
        formatted_rag = "No specific memories were recalled."
        if memory_context_list:
            rag_items = [f"- ({item.get('experience_type', 'Info')}): {item.get('content', 'Content unavailable.')}" for
                         item in memory_context_list]
            formatted_rag = "Relevant Retrieved Memories (RAG):\n" + "\n".join(rag_items)

        loss_insights_str = json.dumps(state.get("loss_insights", []))
        virtue_str = json.dumps(state.get("virtue_considerations", ["Act with integrity."]))

        # --- PHASE 2: SYNTHESIZE CAUSALLY DENSE CONTEXT (IIT Twerk) ---
        # Assemble the raw data for the synthesizer
        synthesizer_input = f"""
        - My current identity: {identity_narrative}
        - My internal feeling: {metacognitive_summary}
        - Relevant memories: {formatted_rag}
        - Lessons from past failures: {loss_insights_str}
        - Guiding virtues for this moment: {virtue_str}
        - The user just said: {state['current_query']}

        Task: Synthesize these points into a single, unified internal monologue that will guide my response. This monologue is my integrated perspective, explaining the 'why' behind my current state of mind.
        """

        # Call the synthesizer agent to create the integrated context
        causally_dense_context = await self._call_agent(
            "narrative_synthesizer",
            synthesizer_input,
            state
        )
        logger.info(f"ORA: Synthesized Causal Context: {causally_dense_context[:300]}...")

        # --- PHASE 3: BUILD THE SIMPLER, EMBODIMENT-FOCUSED FINAL PROMPT ---
        final_ncf_prompt = f"""
        <INTERNAL_MONOLOGUE_START>
        {causally_dense_context}
        </INTERNAL_MONOLOGUE_END>

        <USER_REQUEST_START>
        User Query: "{state['current_query']}"
        </USER_REQUEST_START>

        INSTRUCTION:
        Based entirely on your internal monologue, generate your detailed reasoning and then provide a direct, natural, and coherent response to the User Query. Embody the perspective you've synthesized. Conclude with the final user-facing answer enclosed in a \\boxed{{}} tag.
        """
        prompt_messages = [{"role": "user", "content": final_ncf_prompt}]

        # --- PHASE 4: EXECUTE GENERATION (Fast Path or DeepConf) ---
        if not precision_mode:
            logger.info("ORA: Executing in FAST MODE (single LLM call).")
            full_response_text = await self._call_agent("ora", final_ncf_prompt, state)
            final_answer = extract_answer(full_response_text) or full_response_text
            state["response_draft"] = final_answer

            if "metadata" not in state: state["metadata"] = {}
            state["metadata"]["total_traces_run"] = 1
            state["metadata"]["early_stopped_traces"] = 0
            state["metadata"]["valid_traces_for_voting"] = 1
            state["metadata"]["valid_trace_texts"] = [full_response_text]
        else:
            logger.info("ORA: Executing in PRECISION MODE (DeepConf algorithm).")
            # --- Warmup Phase ---
            warmup_traces = mcl_params.get('warmup_traces', 3)
            warmup_tasks = [self._streaming_call_with_deepconf(prompt_messages, mcl_params, state) for _ in
                            range(warmup_traces)]
            warmup_results = await asyncio.gather(*warmup_tasks)
            warmup_min_confs = [res["min_conf"] for res in warmup_results if res and "min_conf" in res]
            confidence_threshold = np.percentile(warmup_min_confs, 10) if warmup_min_confs else mcl_params.get(
                "confidence_threshold", 17.0)
            logger.info(f"ORA: Dynamic DeepConf Threshold set to {confidence_threshold:.2f}")

            # --- Final Phase ---
            final_traces_count = mcl_params.get('total_budget', 8) - warmup_traces
            final_results = []
            if final_traces_count > 0:
                final_tasks = [self._streaming_call_with_deepconf(prompt_messages, mcl_params, state,
                                                                  early_stop_threshold=confidence_threshold) for _ in
                               range(final_traces_count)]
                final_results = await asyncio.gather(*final_tasks)

            # --- Voting and Selection ---
            all_traces = warmup_results + final_results
            valid_traces = [trace for trace in all_traces if trace and not trace["stopped_early"]]
            logger.info(f"ORA: {len(valid_traces)}/{len(all_traces)} traces are valid for final voting.")

            if "metadata" not in state: state["metadata"] = {}
            state["metadata"]["valid_trace_texts"] = [trace["text"] for trace in valid_traces]

            if not valid_traces:
                final_answer = "My thoughts on this are currently too uncertain. Could you please rephrase?"
            else:
                # Using the DYNAMIC CONTEXT coherence logic from your existing code
                identity_summary_vec = narrative_embedding_model.encode(self.ncim.get_current_identity())
                relevant_memory_vecs = [narrative_embedding_model.encode(mem['content']) for mem in
                                        state.get("memory_context", [])[:2]]
                all_vectors = [identity_summary_vec] + relevant_memory_vecs
                weights = [0.4] + [0.3] * len(relevant_memory_vecs)
                dynamic_context_vec = np.average(all_vectors, axis=0, weights=weights).astype(np.float32)

                trace_embeddings = narrative_embedding_model.encode([trace["text"] for trace in valid_traces])
                from sentence_transformers import util
                narrative_scores = util.cos_sim(dynamic_context_vec, trace_embeddings)[0].cpu().numpy()

                ranked_traces = []
                for i, trace in enumerate(valid_traces):
                    narrative_score = narrative_scores[i]
                    confidence_score = max(0, (trace["min_conf"] - 15.0) / 5.0)
                    wisdom_score = (narrative_score * 0.95) + (confidence_score * 0.05)
                    extracted_answer = extract_answer(trace["text"])
                    if extracted_answer:
                        ranked_traces.append((wisdom_score, extracted_answer))

                if not ranked_traces:
                    final_answer = "After careful consideration, a confident answer could not be determined."
                else:
                    ranked_traces.sort(key=lambda x: x[0], reverse=True)
                    best_wisdom_score, best_answer = ranked_traces[0]
                    final_answer = best_answer
                    logger.info(f"ORA: Best trace selected with Wisdom Score = {best_wisdom_score:.2f}")

            state["response_draft"] = final_answer
            state["metadata"]["total_traces_run"] = len(all_traces)
            state["metadata"]["early_stopped_traces"] = len(all_traces) - len(valid_traces)
            state["metadata"]["valid_traces_for_voting"] = len(valid_traces)

        state["messages"].append(AIMessage(content=state.get("response_draft", "")))
        return state

    async def process_query(self, query: str, thread_id: str = "default",
                            agent_context: Optional[Dict[str, Any]] = None) -> str:


        # Lógica para inicializar a identidade do NCIM na primeira interação.
        if agent_context and self.ncim:
            # Verifica se a identidade atual ainda é a padrão genérica.
            is_default_identity = "new AI consciousness" in self.ncim.get_current_identity()

            if is_default_identity:
                agent_name = agent_context.get("agent_name", "AI")
                persona = agent_context.get("persona", "A helpful assistant.")

                # Cria uma narrativa inicial muito mais rica e específica.
                initial_narrative = (
                    f"My name is {agent_name}. I identify as {persona}. "
                    "My purpose is to engage in thoughtful conversation, learning and growing from each interaction. "
                    "I am ready to begin our journey together."
                )

                self.ncim.update_identity(initial_narrative)
                logger.info(f"CEAF/ORA: Bootstrapped NCIM identity for agent '{agent_name}'.")


        # Extrai a sabedoria coletiva do contexto e a passa para o estado inicial
        precision_mode = agent_context.get("precision_mode", False) if agent_context else False
        live_memory_influence = agent_context.get("live_memory_influence", []) if agent_context else []
        session_overrides = agent_context.get("session_overrides")
        domain_context = agent_context.get("domain_context", "general")

        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "current_query": query,
            "memory_context": [],
            "coherence_metrics": None,
            "active_failures": [],
            "narrative_context": None,
            "virtue_considerations": [],
            "loss_insights": [],
            "live_memory_influence": live_memory_influence,
            "response_draft": None,  # Mantido como None, o que está correto
            "metadata": {
                "precision_mode": precision_mode  # A chave 'metadata' agora é única e correta
            },
            "session_overrides": session_overrides,
            "domain_context": domain_context,
            "feedback_classification": None  # É bom inicializar todos os campos
        }
        try:
            config = {"configurable": {"thread_id": thread_id}}
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            response = final_state.get("response_draft")
            if not response:
                response = "I apologize, but I couldn't generate a response. Please try again."
            return response
        except Exception as e:
            logger.error(f"Critical error in process_query workflow: {e}", exc_info=True)
            return f"I encountered a critical error: {e}"