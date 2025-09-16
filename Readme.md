
# Aura AI Agents: Advanced Multi-Agent AI Framework

Online DEMO: https://cognai.space/


[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)



**Aura AI Agents** is a powerful, flexible, and scalable framework for building, managing, and interacting with advanced conversational AI agents. Built on Python with FastAPI, it provides a robust foundation for creating multi-tenant AI experiences, where each agent operates in an isolated environment with its own memory system, personality, and capabilities.

Similar to Character.AI but with complex prompt engineering and memory systems. 

FrontEnd Index
<img width="1681" height="912" alt="image" src="https://github.com/user-attachments/assets/966b81c1-f516-473c-ba0b-fe20fa5aacf0" />

The framework is built around two distinct and powerful agent architectures: **NCF (Narrative Context Framing)** and the cutting-edge experimental **CEAF (Coherent Emergence Agent Framework)**.

## ✨ Key Features

- **Dual Agent Architectures:**
  - **NCF (Narrative Context Framing):** A robust, production-ready system that uses sophisticated prompt engineering, RAG (Retrieval-Augmented Generation), and a reflection loop to create agents with deep context and personality.
  - **CEAF (Coherent Emergence Agent Framework):** An experimental autonomous agent architecture that utilizes a metacognitive control loop (MCL) and an adaptive memory architecture (AMA) to develop a coherent, emergent identity over time.

- **Advanced Memory Systems:**
  - **MemoryBlossom:** An isolated memory system for each NCF agent, featuring distinct memory types, salience decay, and emotional scores.
  - **Adaptive RAG:** An optional layer over MemoryBlossom that adds adaptive concept clustering and domain specialization for even smarter memory retrieval.
  - **AMA (Adaptive Memory Architecture):** The core of the CEAF system, which autonomously organizes experiences into concept clusters.

- **Multi-Tenant & User-Isolated:** The architecture is designed from the ground up to serve multiple users, ensuring each agent's data, memories, and files remain completely isolated and secure.

- **Extensible Specialist Agents:** Offload complex tasks to dedicated specialist agents. The system includes out-of-the-box examples for image generation (Tensor.Art) and speech processing (Hugging Face).

- **Comprehensive REST API:** A robust API built with FastAPI provides endpoints for everything from user authentication (JWT) and agent management to chat interaction, memory management, and file uploads.

- **Agent Marketplace & Cloning:** Users can publish their private agents as "public templates" to the marketplace. Other users can then clone these agents, inheriting their foundational personality and memories to kickstart their own experience.

- **Per-Agent Integrated RAG:** Users can upload files (PDFs, TXT) to an agent's personal storage. The agent can then search these documents to answer questions, giving each agent its own custom knowledge base.

- **Live Collective Memory:** An optional shared wisdom system where agents can contribute anonymized insights, allowing other agents to benefit from collective knowledge.

- **Dynamic Credit System:** A built-in billing system that deducts credits based on the LLM model used, with configurable costs for different model tiers.

- **Precision Mode:** Measure logprobs, run up to 8 paralel llm calls, measure confidence and retrieve higher values.  Based on DeepConf.

- **Specialist Agentse:** As for today, Aura has 2 Specialist Agents: Tensor.Art and HugginFace. Any Agent in Aura can call those. You can add more check extended documentation. 

## 🏛️ Extended Documentation 

- Guide to the CEAF Governor Module : https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/blob/main/ReadMeCEAF.md
- Guide to the NCF Governor Module: https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/blob/main/ReadmeNCF.md
- Guide to Precision Mode(DeepConf) in CEAF: https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/blob/main/ReadMeDeepConf.md 
- Guide to the FrontEnd : https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/blob/main/ReadmeIndexFrontEnd.md
- Guide to Models and Credit: https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/blob/main/ReadmeMODELS.md
- Guide to the MemoryBlossom and Adaptive Memory: https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/blob/main/ReadmeMemory.md
- Guide How to Add Specialist Agents: https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/blob/main/ReadmeAddSpecialistAgents.md
- Guide How to Add MCP/A2A Specialists Agents: https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/blob/main/ReadMeMCPA2A.md
- Guide How to Biograhical Memories: https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/blob/main/ReadmeBiographicalMemories.md

## 🏛️ Architecture Overview

The AuraCode system is built in layers to ensure modularity and scalability:

1.  **API Layer (FastAPI - `api/routes.py`):** The entry point for all interactions. It handles HTTP requests, user authentication (JWT), and data validation.
2.  **Management Layer (`agent_manager.py`):** The heart of the system. It orchestrates the creation, deletion, and retrieval of agent instances. It decides whether to instantiate an NCF agent or a CEAF adapter.
3.  **Agent Layer (NCF/CEAF Instances):**
    - **NCFAuraAgentInstance:** The standard agent implementation. It utilizes `ncf_processing.py` to build context-rich prompts and interacts with MemoryBlossom.
    - **CEAFAgentAdapter:** An adapter layer that presents the complex CEAF system with a compatible interface, allowing it to be used transparently by the rest of the system.
4.  **Memory Layer (`memory_system/` & `ceaf_system/AMA.py`):** Handles the storage, retrieval, and management of each agent's knowledge. Includes MemoryBlossom, Adaptive RAG, and CEAF's Adaptive Memory Architecture.
5.  **Persistence Layer (SQLAlchemy & Filesystem):**
    - A SQLite database (`aura_agents.db`) stores user and agent metadata.
    - The `agent_storage/` directory contains each agent's configuration, memory, and RAG files in an isolated folder structure.

## 🧠 Core Concepts: NCF vs. CEAF

| Feature | NCF (Narrative Context Framing) | CEAF (Coherent Emergence Agent Framework) |
| :--- | :--- | :--- |
| **Approach** | Sophisticated Prompt Engineering | Autonomous, Self-Organizing System |
| **Awareness** | Prompt-based; context is provided on each turn. | Emergent; internal state evolves over time. |
| **Components** | `ncf_processing`, `MemoryBlossom`, `AuraReflector` | `AMA`, `MCL`, `ORA`, `NCIM`, `VRE`, `AURA` |
| **Ideal for** | Robust character agents, context-aware assistants. | AGI research, consciousness simulation, agents that learn from failure. |
| **Complexity** | Moderate | High (Experimental) |

## 🚀 Getting Started

Follow these steps to set up and run the AuraCode server locally.

### 1. Prerequisites

- Python 3.10+
- langChain
- LangGraph
- Google ADK
- OpenRouter
- LiteLLM

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/tree/main/AuraCode
    or Download https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/blob/main/AuraCode.zip
    cd AuraCode 
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The project uses several libraries. Install them using:
    ```bash
    pip install -r requirements.txt 
    Double check
    ```

4.  **Configure Environment Variables:**
    Create a file named `.env` in the project root and add your API keys: - CHECK Example.ENV ********* 
    ```env
    # Required for most LLM operations
    OPENROUTER_API_KEY="sk-or-v1-..."

    # Key for the Image Generation Specialist Agent
    TENSORART_API_KEY="..."
    
    # Key for the Speech Specialist Agent
    HUGGINGFACE_API_KEY="..."

    # Secret key for JWT authentication tokens (change to something secure)
    JWT_SECRET_KEY="a-very-long-and-secure-secret-key"
    ```

5.  **Set up the Database:**
    Run the init or migration script to create and verify the SQLite database schema.
    ```bash
    python init_db.py
    or 
    python db_migration.py
    ```

### 3. Running the Application

With the virtual environment activated and the `.env` file configured, start the API server:

```bash
uvicorn main_app:app --host 0.0.0.0 --port 8000 --reload
```

- The API will be available at `http://localhost:8000`.
- Interactive API documentation (Swagger UI) will be at `http://localhost:8000/docs`.
- If the frontend is present in the `frontend/` folder, it will be served at `http://localhost:8000/`.

## 🛠️ Scripts & Utilities

The project includes several scripts in the root directory for management and testing:

- `db_migration.py`: Safely applies schema changes to the SQLite database.
- `uploader.py`: A CLI tool for bulk-creating agents from JSON "biography" files. (check  https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/blob/main/AuraCode/Documentation/Important%20CURLS.txt ) 
- `wisdom_collector.py`: A process that can be run in the background to index insights from the "Live Collective Memory."
- `check_db_agents.py`: A CLI tool to check the publication status of agents in the database.
- `ExportCodeBaseToTXT.py`: Export all .py files into a single .txt file 
- `check_db_agents.py`: Check if Agent exists


## 📁 Frontend 

- It has a simple HTML, CSS, JS frontend in the /frontend folder 
- i'm also uploading a very deprecated Android APP that serves as a frontEnd , but it is very deprecated but surely can be used as a base... check https://github.com/IhateCreatingUserNames2/Aura_AI_Agents/blob/main/srcAndroidFront.zip

## 📁 Project Structure (Simplified)

```
AuraCode/
├── agent_storage/           # Isolated storage for agent data
├── api/
│   └── routes.py            # FastAPI API endpoints
├── ceaf_system/             # Components for the experimental CEAF framework
│   ├── AMA.py               # Adaptive Memory Architecture
│   ├── MCL.py               # Metacognitive Control Loop
│   ├── ORA.py               # Orchestrator/Responder Agent (LangGraph)
│   ├── ...                  # Other components (NCIM, VRE, AURA)
├── database/
│   └── models.py            # SQLAlchemy data models
├── memory_system/
│   ├── memory_blossom.py    # Main memory system for NCF agents
│   ├── memory_models.py     # Data model for a single memory
│   └── ...
├── agent_manager.py         # Core logic for managing agent instances
├── enhanced_memory_system.py # Adaptive RAG implementation
├── main_app.py              # Entry point to start the uvicorn server
├── ncf_processing.py        # Logic for building the NCF prompt
├── prebuilt_agents_system.py # System for creating and managing pre-built agents
├── rag_processor.py         # Logic for processing and searching files (RAG)
├── db_migration.py          # Database migration script
├── .env.example             # Example file for environment variables
└── ...
```

## 🤝 Contributing

Contributions are welcome! If you'd like to improve AuraCode, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature (`git checkout -b feature/my-amazing-feature`).
3.  Commit your changes (`git commit -m 'Add my amazing feature'`).
4.  Push to the branch (`git push origin feature/my-amazing-feature`).
5.  Open a Pull Request.

## 📄 License

This project is released under a custom license designed to be free for small-scale use. 

- If you need help find me on : discord https://discord.gg/Urb27E9C

