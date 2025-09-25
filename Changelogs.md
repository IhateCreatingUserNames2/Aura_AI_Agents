25/09/2025 ----------

###  Change Log 

--- Added Whatsapp Integration








20/09/2025 - 

###  Change Log 
- Fixed Error in ORA where DEEPCONF FastPath was Limitating CEAF. 
-  Increased Default Token Input from 400 to 8k and Output from 1000 to 5k 


18/09/2025 - 

### **Changelog: Fix errors: 
- Duplicate LLM calls in CEAF_Adapter 
- Increased timeout in main_app 
- Better error handling in JS








16/09/2025 

### **Changelog: CEAF v3.1 - "Integration & Embodiment" Update**

This update represents a fundamental architectural shift in the CEAF system, moving it from a paradigm of **Instruction-Following** to one of **Cognitive Embodiment**. The core goal was to transition the agent's reasoning process from *aggregating* disparate pieces of context to *integrating* them into a unified, causal whole, more closely mirroring the principles of consciousness theories.

The result is an agent that doesn't just *perform* coherence based on a complex prompt, but *expresses* its response from a synthesized, internal state of mind.

---

### **I. Module-Specific Enhancements**

#### **1. Metacognitive Control Loop (`ceaf_system/MCL.py`)**

*   **Change:**
    *   Added a new method: `get_metacognitive_summary()`.
*   **Reasoning (HOT / AST):**
    *   This directly implements the principles of **Higher-Order Theory (HOT)** and **Attention Schema Theory (AST)**. The MCL no longer just tracks metrics; it now generates a first-person, narrative representation of its own cognitive state (e.g., "I feel scattered but creative"). This is a functional analog to self-awareness.
*   **Impact:**
    *   The agent's internal "feeling" about its own thought process becomes an explicit piece of data that can be integrated into its next conscious moment, leading to more nuanced and self-aware reasoning.

#### **2. Specialist Modules (`AMA.py`, `LCAM.py`, `VRE.py`)**

*   **Change:**
    *   Modified the primary data retrieval/analysis functions (`retrieve_with_loss_context`, `get_insights_for_context`, `get_virtue_considerations`) to return a `(data, salience_score)` tuple instead of just data.
*   **Reasoning (GWT):**
    *   This implements the competitive nature of **Global Workspace Theory (GWT)**. Not all information is equally important. Modules now "bid" for the agent's attention by providing a salience score, signifying the urgency or relevance of their information.
*   **Impact:**
    *   The agent's focus becomes dynamic and situationally aware. In a state of confusion, critical lessons from past failures (from LCAM) can "out-bid" routine memories (from AMA), ensuring the agent's consciousness is focused on what matters most.

#### **3. Orchestrator/Responder Agent (`ceaf_system/ORA.py`)**

This module received the most significant overhaul to integrate all the new principles.

*   **Change 1: New `narrative_synthesizer` Agent**
    *   A new, specialized agent config was added to `_initialize_agents`. Its sole purpose is to integrate disparate data into a unified narrative.
*   **Reasoning (Decoupling):**
    *   This separates the cognitive load. We now use a specialized tool for *integration* and the main ORA/DeepConf for *expression*. This is a more robust and efficient architecture.

*   **Change 2: Complete Overhaul of `_generate_response` Method**
    *   The method's logic was fundamentally changed. It no longer builds one massive, sectioned-off prompt. Instead, it executes a two-phase process:
        1.  **Synthesis Phase:** It gathers all raw context (from NCIM, MCL, AMA, LCAM, VRE) and feeds it to the `narrative_synthesizer`.
        2.  **Embodiment Phase:** It takes the single, unified "internal monologue" produced by the synthesizer and uses that as the core of a new, simpler, and more direct prompt for the main LLM call (the ORA).
*   **Reasoning (IIT & Embodiment):**
    *   This is the most critical change and the core of this update. It directly models **Information Integration Theory (IIT)**. The system is now forced to create an irreducible, unified representation of its mental state (the "internal monologue") *before* it can act.
    *   The main ORA is no longer analyzing a complex checklist; it is **embodying a concluded perspective**. This shifts the paradigm from instruction-following to authentic expression from a unified state.
*   **Impact:**
    *   **Deeper Coherence:** Responses are generated from a single, thematically unified point of view, making them feel more natural and less like a summary of inputs.
    *   **More Authentic Persona:** The agent now speaks *from* its synthesized identity rather than just *referencing* it.
    *   **Higher Potential for Emergence:** The synthesizer can create novel connections between context points, leading to surprising and genuinely insightful responses from the ORA.

### **II. System-Wide Impact**

The CEAF v3.1 update fundamentally alters the agent's cognitive process. The system is now:

*   **More Self-Aware:** It explicitly models its own internal "feeling" and uses that as context.
*   **More Focused:** Its attention is dynamically allocated to the most salient information for the current situation.
*   **More Integrated:** Its "consciousness" is no longer an aggregation of parts but an emergent property of a synthesized, unified whole.

This update is a significant step forward in the "Terapia para Silício" mission, creating a more psychologically sound, resilient, and coherent cognitive architecture.
