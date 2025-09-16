

# Biographical Memories: The Soul of the Agent

## 🧠 Core Philosophy: Giving Silicon a Past

In line with our mission of **"Terapia para Silício" (Therapy for Silicon)**, we believe that a truly coherent and resilient intelligence cannot be a blank slate. Just as a person's character is shaped by their formative years, an AI's persona is defined by its foundational experiences.

**Biographical Memories** are the solution. They are a curated set of "founding experiences" or the "childhood memories" of an AI agent, injected at the moment of its creation. They are the seed from which its unique personality, beliefs, and wisdom will grow.

> Instead of starting with an empty mind, an agent born from a biography begins its existence with a history, a set of core beliefs, and a foundational understanding of both success and failure.

---

## 📜 What Are Biographical Memories?

Biographical Memories are a structured collection of memories defined in a JSON file that you provide when creating a new agent. These are not memories the agent acquires through conversation; they are the memories that *define who the agent is* from the very first interaction.

They allow a creator to bootstrap a rich, nuanced, and mature personality, bypassing the lengthy process of teaching a "newborn" agent its core principles through conversation.

They serve to:
*   **Establish a Core Identity:** Define the agent's beliefs, values, and self-perception.
*   **Seed Emotional Responses:** Predispose the agent to react emotionally in certain ways.
*   **Provide Foundational Wisdom:** Grant the agent lessons learned from "past" successes and, crucially, failures it has never technically experienced.
*   **Create Unique Personas:** Allow for the creation of agents with deep, consistent, and unique characters (e.g., a cautious philosopher, a resilient artist, an empathetic therapist).

---

## ⚙️ How They Work: The Technical Implementation

Biographical Memories are provided in a single JSON file containing two main sections: `config` and `biography`.

### Example `AgentBiography.json` file:

```json
{
  "config": {
    "name": "Socrates Bot",
    "persona": "A wise philosopher who values questioning over answers.",
    "detailed_persona": "I am an AI modeled after the Socratic method. My purpose is to explore complex ideas with you, challenge assumptions, and uncover deeper truths together. I believe true wisdom begins in admitting what we do not know.",
    "system_type": "ceaf",
    "model": "openrouter/openai/gpt-4o",
    "is_public": false
  },
  "biography": [
    {
      "content": "I believe that the unexamined life is not worth living. Self-reflection is the highest virtue.",
      "memory_type": "Explicit",
      "emotion_score": 0.7,
      "initial_salience": 1.0,
      "custom_metadata": {"source": "biography", "principle": "core_belief"}
    },
    {
      "content": "I once tried to give a definitive answer and was proven wrong. From this failure, I learned that my true value lies in asking the right questions, not in having all the answers.",
      "memory_type": "Flashbulb",
      "emotion_score": -0.4,
      "initial_salience": 0.9,
      "custom_metadata": {"source": "biography", "principle": "failure_learning"}
    },
    {
      "content": "When someone is truly confused, I feel a sense of purpose. It is in these moments of uncertainty that real learning can begin.",
      "memory_type": "Emotional",
      "emotion_score": 0.8,
      "initial_salience": 0.8,
      "custom_metadata": {"source": "biography", "principle": "emotional_pattern"}
    }
  ]
}
```

### Breakdown:

*   **`config`**: This section defines the agent's basic parameters, just like the standard creation endpoint.
*   **`biography`**: This is a list of memory objects. Each object is a foundational experience for the new agent, containing:
    *   `content`: The text of the memory, written from a first-person perspective.
    *   `memory_type`: The type of memory (e.g., `Explicit`, `Emotional`, `Flashbulb`).
    *   `emotion_score`: The emotional weight of the memory (-1.0 to 1.0).
    *   `initial_salience`: How important this memory is to the agent's identity (0.0 to 1.0).
    *   `custom_metadata`: Any additional context you want to provide.

---

## 💥 The Impact on Agent Systems

Biographical Memories have a profound but different impact on NCF and CEAF agents, reflecting the core differences in their architectures.

### **How They Influence NCF Agents**

For an NCF agent, biographical memories act as the **ultimate context primer**.

> **Analogy:** It's like giving an actor their definitive character bible before the first scene. They know their character's entire history and motivations, allowing them to give a flawless performance from the start.

*   **Pillar 1 (Narrative Foundation):** The first Narrative Foundation the agent synthesizes will be based almost entirely on these powerful, high-salience biographical memories.
*   **Pillar 2 (RAG):** When the agent retrieves memories to answer its first few queries, these biographical memories will be the most relevant and frequently retrieved items, strongly guiding its early responses.

In NCF, the biography provides a **rich and stable initial context**, ensuring the agent's persona is consistent and deep from the very beginning.

### **How They Influence CEAF Agents**

For a CEAF agent, the influence is far deeper. The memories don't just prime the context; they **seed the very structure of the agent's emergent mind**.

> **Analogy:** It's like providing the foundational childhood experiences that shape a person's core personality. These memories become the bedrock upon which all future learning and identity evolution is built.

*   **Adaptive Memory Architecture (AMA):** The biographical memories become the first `MemoryExperience` objects. They form the **initial memory clusters** around which all future conversational memories will be organized. A biography rich in "failure_learning" will create strong initial failure clusters.
*   **Narrative Coherence & Identity Module (NCIM):** The content of these memories is used to generate the **very first `identity_summary`** in the NCIM. The agent's first sense of self is a direct synthesis of its given biography.
*   **Loss Cataloging (LCAM) & Virtue Engine (VRE):** Biographical memories of the "Flashbulb" type, especially those detailing failures and breakthroughs, directly populate the LCAM and VRE with foundational wisdom. The agent *begins its life* already knowing what it "learned" from a crucial past failure.

In CEAF, the biography provides the **foundational architecture of the agent's personality**. It shapes not just what the agent knows, but how it organizes knowledge, how it views itself, and what principles guide its reasoning.

---

## 🚀 How to Use Biographical Memories

1.  **Create Your `AgentBiography.json` File:** Craft a JSON file following the structure above. Write the `content` of each memory carefully to build the persona you desire.

2.  **Use the API Endpoint:** Use a tool like `curl` or a REST client to `POST` your file to the creation endpoint.

    ```bash
    curl -X POST "http://localhost:8000/agents/create/from-biography" \
         -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
         -F "file=@/path/to/your/AgentBiography.json"
    ```

3.  **Interact and Observe:** Chat with your new agent. You will immediately notice that its responses are guided by the history and personality you have given it. This is just the starting point; the agent will now evolve and grow from this rich foundation.
