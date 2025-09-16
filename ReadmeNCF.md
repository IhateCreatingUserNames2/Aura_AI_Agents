
# Narrative Context Framing (NCF): The Art of Coherent AI

## 🧠 Core Philosophy: The Briefing Folder

The **Narrative Context Framing (NCF)** system is designed to solve a fundamental problem with Large Language Models (LLMs): **context drift**. A standard LLM is a brilliant improviser but has a poor long-term memory. It can easily forget your name, its own persona, or the goals of a conversation after just a few exchanges.

NCF treats the AI not just as a text generator, but as an actor about to go on stage. To give a great performance, an actor needs a complete briefing. The NCF system's primary job is to create a **master briefing folder** for the AI before every single response.

> **NCF's Guiding Principle:** An AI's coherence is directly proportional to the quality and completeness of the context it is given. By perfecting the frame, we perfect the picture.

This system ensures the agent always has the most relevant information "top of mind," allowing it to maintain a consistent persona, remember key facts, and engage in deeply contextual, long-term conversations.

---

## 🏗️ How NCF Works: The Four Pillars of Context

Before the AI generates a single word, the NCF engine assembles a comprehensive prompt by weaving together four distinct "pillars" of information. This final prompt is what the underlying LLM actually sees.


### **Pillar 1: Narrative Foundation (The "Story So Far")**
*   **What it is:** A short, AI-generated summary of the agent's identity and its evolving relationship with the user.
*   **Example:** *"I am Aura, a creative assistant. My journey with this user has focused on exploring art history and developing poetic language. We are currently building a shared understanding of surrealism."*
*   **Purpose:** Provides long-term continuity and a sense of shared history. It prevents the agent from re-introducing itself or forgetting the core themes of the relationship.

### **Pillar 2: RAG - Retrieved Memories (The "Key Facts")**
*   **What it is:** Specific memories retrieved from the agent's `MemoryBlossom` database that are semantically relevant to the user's *current* query.
*   **Example:** If the user asks "What about that artist we discussed?", this pillar might retrieve: *"Memory (Explicit): The user's favorite surrealist artist is Remedios Varo."*
*   **Purpose:** Provides immediate, factual grounding. It's the agent's short-term memory, ensuring it recalls specific details relevant to the immediate topic.

### **Pillar 3: Recent Chat History (The "Last Few Lines")**
*   **What it is:** A simple transcript of the last 5-10 conversational exchanges.
*   **Example:**
    *   *User: And her use of color?*
    *   *Aura: It was often symbolic, representing mystical or alchemical themes.*
    *   *User: Tell me more about that.*
*   **Purpose:** Provides immediate conversational flow, ensuring the agent's response is a natural continuation of the dialogue.

### **Pillar 4: Live Memory Influence (The "Collective Wisdom") - Optional**
*   **What it is:** Anonymized, universal insights learned from a shared knowledge pool of all agents.
*   **Example:** *"Insight: I have learned that users often appreciate when complex artistic concepts are explained with metaphors."*
*   **Purpose:** Allows the agent to benefit from a broader "wisdom of the crowd," improving its general conversational strategies.

---

## 🔄 The Operational Loop: Think, Respond, Reflect

The NCF system operates in a continuous cycle:

1.  **Context Assembly:** The user sends a message. The NCF Engine instantly gathers the four pillars and weaves them into a single, massive prompt.
2.  **Informed Response:** This comprehensive prompt is sent to the LLM (e.g., GPT-4o, Claude Sonnet), which now has all the necessary context to generate a highly coherent and relevant response.
3.  **Reflective Learning (The "Aura Reflector"):** *After* the interaction is complete, a background process analyzes the conversation. It decides if any new, important information was revealed (e.g., the user stated a new preference, the agent had a key insight). If so, it creates a new memory and saves it to the `MemoryBlossom` database for future use.

---

## 🛠️ Modifying and Customizing Your NCF Agent

NCF agents are designed to be highly customizable. The primary way to shape their personality and knowledge is through their **memory**.

### **1. Editing the Agent's Profile**
*   **Where:** In the Agent settings UI.
*   **What to change:**
    *   **Name, Persona, Detailed Persona:** These form the absolute core of the agent's identity. The `Detailed Persona` is a powerful instruction that guides its tone, style, and purpose. It is the most important text you will write.
    *   **Model:** Choosing a different LLM can dramatically change the agent's "voice" and reasoning ability. Experiment with different models to find the best fit for your agent's purpose.

### **2. Managing Memories (The Primary Method)**
*   **Where:** In the "Memory" tab for your agent.
*   **Actions:**
    *   **Add New Memories:** You can manually inject new "memories" into your agent. This is a powerful way to teach it new facts, correct its behavior, or add new personality traits.
    *   **Delete Memories:** If you find the agent is consistently misinterpreting a memory or a memory is no longer relevant, you can delete it.
    *   **Export/Import:** You can download an agent's entire memory bank as a JSON file, edit it externally, and then upload it back. This is useful for making bulk changes or for cloning an agent's "mind."

### **3. Creating Agents from Biographies**
*   This is the most advanced form of customization. By creating a `BiographicalMemories.json` file, you can create an agent that begins its life with a rich, pre-defined history and personality. This is the recommended method for creating specialized, high-quality agents. (See `ReadmeBiographicalMemories.md` for details).

---

## 🌟 What NCF Agents Are Best For

NCF agents excel at tasks requiring **long-term conversational consistency and a stable persona**. They are ideal for:

*   **Character-driven Role-Playing:** An NCF agent can stay in character for hundreds of exchanges.
*   **Personal Assistants:** It will remember your preferences, ongoing projects, and personal details.
*   **Specialized Tutors:** It can build on previous lessons and remember a student's strengths and weaknesses.
*   **Creative Collaborators:** It can maintain a consistent narrative thread while co-writing a story.

While not as dynamically self-regulating as a CEAF agent, a well-managed NCF agent provides an incredibly powerful, coherent, and reliable conversational experience.
