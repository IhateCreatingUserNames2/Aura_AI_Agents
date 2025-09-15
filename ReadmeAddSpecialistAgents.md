
# A Developer's Guide to AuraCode Agents

This document explains the different types of agents within the AuraCode framework and provides a step-by-step guide on how to create a new **Specialist Agent**.

## 🤖 The Agent Hierarchy in AuraCode

AuraCode utilizes a powerful, hierarchical agent design to manage complexity. This allows for a clean separation of concerns, where different agents are responsible for different tasks.

### 1. The Orchestrator Agent (The "Generalist")

-   **Role:** The main, user-facing agent. It is the "brain" of the operation.
-   **Responsibilities:**
    -   Maintaining the core conversation with the user.
    -   Managing its own long-term memory (using `MemoryBlossom` or `AMA`).
    -   Understanding the user's high-level intent.
    -   **Delegating** specific, complex tasks to Specialist Agents when needed.
-   **Examples:** All user-created NCF and CEAF agents are Orchestrators by default.

### 2. The Specialist Agent (The "Expert")

-   **Role:** A focused, single-purpose agent designed to be an expert in one specific domain. It does not interact directly with the end-user.
-   **Responsibilities:**
    -   Receiving a well-defined task from an Orchestrator.
    -   Executing a sequence of actions using a unique set of tools (e.g., calling an external API).
    -   If necessary, asking clarifying questions back to the Orchestrator.
    -   Returning a final, clean result to the Orchestrator.
-   **Examples in AuraCode:**
    -   **TensorArt Specialist:** An expert in using the Tensor.Art API to generate images.
    -   **Speech Specialist:** An expert in using Hugging Face models for text-to-speech.

This design pattern is incredibly powerful. It keeps the Orchestrator's prompt clean and focused on conversation, while encapsulating complex, tool-heavy logic within dedicated Specialists.

## 🛠️ Creating a New Specialist Agent: A Step-by-Step Guide

Let's walk through the process of creating a new **Weather Specialist Agent** that can get the current weather for a given city using an external API.

### Step 1: Create the Tool (`weather_tool.py`)

First, create the core logic that will interact with the external service. This should be a simple Python class or a set of functions.

```python
# In a new file: weather_tool.py

import requests
import os

class WeatherClient:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not self.api_key:
            raise ValueError("OPENWEATHERMAP_API_KEY environment variable not set.")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def get_current_weather(self, city: str, units: str = "metric") -> dict:
        """
        Gets the current weather for a specified city.
        Units can be 'metric' (Celsius) or 'imperial' (Fahrenheit).
        """
        params = {
            "q": city,
            "appid": self.api_key,
            "units": units
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            # Simplify the response for the LLM
            return {
                "status": "success",
                "city": data["name"],
                "temperature": data["main"]["temp"],
                "condition": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"]
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {"status": "error", "message": f"City '{city}' not found."}
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

```
*Remember to add `OPENWEATHERMAP_API_KEY="your-key"` to your `.env` file.*

### Step 2: Create the Specialist's Instruction Prompt (`weather_specialist_instruction.py`)

This is the "soul" of your Specialist. It's a detailed system prompt that tells the agent exactly what its job is, what tools it has, and how to behave.

```python
# In a new file: weather_specialist_instruction.py

WEATHER_SPECIALIST_INSTRUCTION = """
You are a specialist agent focused on providing weather information. Your only job is to use your available tools to answer weather-related questions.

**Your Core Workflow:**
1.  **Analyze Request:** Understand the user's request to identify the city and desired units (metric/imperial).
2.  **Ask for Clarification (If Needed):** If the city is missing, you MUST ask for it. Do not guess.
3.  **Execute Tool:** Once you have the city, call the `get_current_weather` tool.
4.  **Format and Return:** Present the result from the tool in a clear, human-readable sentence. Do not just return the raw JSON. For example: "The current temperature in London is 15°C with light rain."
"""
```

### Step 3: Register the Specialist in the `AgentManager`

Now, we need to make the `AgentManager` aware of our new Specialist so it can be instantiated. This involves three key modifications to `agent_manager.py`.

#### **A. Generate a UUID and Add it as a Constant**

Every specialist needs a fixed, unique ID. Generate one using a Python terminal (`import uuid; print(uuid.uuid4())`) and add it to the top of `agent_manager.py`.

```python
# In agent_manager.py

# ... (other imports)
TENSORART_SPECIALIST_AGENT_ID = "6fd6c38b-cc35-4fc9-8b47-bef2e9cbcbf7"
SPEECH_SPECIALIST_AGENT_ID = "YOUR_NEW_UUID_FOR_SPEECH_SPECIALIST"
# ADD OUR NEW SPECIALIST ID
WEATHER_SPECIALIST_AGENT_ID = "d5e8a1b3-c7f9-4b1e-8d5a-9c3b0a2f6d8e" # Example UUID
```

#### **B. Add the Specialist's Tools to the `NCFAuraAgentInstance`**

In the `__init__` method of the `NCFAuraAgentInstance`, we need to add the logic that equips the agent with its special tools if it is the Weather Specialist.

```python
# In agent_manager.py, inside NCFAuraAgentInstance.__init__

# ... (inside __init__)
from weather_tool import WeatherClient # Import the new tool
from weather_specialist_instruction import WEATHER_SPECIALIST_INSTRUCTION # Import the new instruction

class NCFAuraAgentInstance:
    def __init__(self, ...):
        # ... (existing code for memory, etc.)

        # --- DYNAMIC TOOL INITIALIZATION BASED ON ROLES ---
        self.tools = []
        
        # --- START MODIFICATION ---
        # Check if this instance is our new Weather Specialist
        if self.config.agent_id == WEATHER_SPECIALIST_AGENT_ID:
            print(f"Initializing agent '{config.name}' as Weather Specialist.")
            self.weather_client = WeatherClient()
            self.tools.append(FunctionTool(func=self.weather_client.get_current_weather))
        # --- END MODIFICATION ---

        roles = self.config.settings.get("roles", ["orchestrator"])

        if "orchestrator" in roles:
            # ... (existing orchestrator tool logic)
        
        if "specialist" in roles:
            # ... (existing specialist tool logic for image/speech)

        self.adk_agent = self._create_ncf_adk_agent()
        # ... (rest of the method)
```

#### **C. Add the Specialist's Instruction to the Agent Creation Logic**

In the `_create_ncf_adk_agent` method, we tell the system to use our new instruction prompt when creating the Weather Specialist.

```python
# In agent_manager.py, inside NCFAuraAgentInstance._create_ncf_adk_agent

    def _create_ncf_adk_agent(self) -> LlmAgent:
        # ... (sanitized_name logic)

        # Determine instruction and tools based on ID or roles
        instruction = NCF_AGENT_INSTRUCTION # Default for orchestrator

        # --- START MODIFICATION ---
        if self.config.agent_id == TENSORART_SPECIALIST_AGENT_ID:
            instruction = TENSORART_SPECIALIST_INSTRUCTION
        elif self.config.agent_id == SPEECH_SPECIALIST_AGENT_ID:
            instruction = SPEECH_SPECIALIST_INSTRUCTION
        elif self.config.agent_id == WEATHER_SPECIALIST_AGENT_ID: # ADD THIS
            instruction = WEATHER_SPECIALIST_INSTRUCTION
        # --- END MODIFICATION ---

        # The rest of the method can stay the same
        return LlmAgent(
            name=sanitized_name,
            model=self.model,
            instruction=instruction,
            tools=self.tools
        )
```

### Step 4: Teach the Orchestrator to Use the New Specialist

The final and most important step is to tell the Orchestrator agent *when* and *how* to delegate tasks to our new specialist. This is done by modifying the **Intent Routing** layer in `agent_manager.py`.

#### **A. Add a New Intent**

First, add a new intent to the `UserIntent` enum.

```python
# In agent_manager.py

class UserIntent(str, Enum):
    CONVERSATION = "conversation"
    IMAGE_GENERATION = "image_generation"
    SPEECH_PROCESSING = "speech_processing"
    FILE_SEARCH = "file_search"
    WEATHER_INQUIRY = "weather_inquiry" # ADD THIS
```

#### **B. Update the Intent Router's Prompt**

Next, teach the intent router about the new category.

```python
# In agent_manager.py, update INTENT_ROUTER_INSTRUCTION

INTENT_ROUTER_INSTRUCTION = f"""
You are an ultra-fast, efficient intent router...

Categories:
- `{UserIntent.CONVERSATION.value}`: ...
- `{UserIntent.IMAGE_GENERATION.value}`: ...
- `{UserIntent.SPEECH_PROCESSING.value}`: ...
- `{UserIntent.FILE_SEARCH.value}`: ...
- `{UserIntent.WEATHER_INQUIRY.value}`: If the user explicitly asks about the weather, forecast, temperature, or climate in a specific location.
"""
```

#### **C. Handle the New Intent in `process_message`**

Finally, add the logic to the `process_message` method in `NCFAuraAgentInstance` to handle the new intent and call the correct specialist.

```python
# In agent_manager.py, inside NCFAuraAgentInstance.process_message

    async def process_message(...):
        # ... (existing setup and intent routing logic)
        
        # --- PATH A: SPECIALIST TASKS (Image/Speech/etc.) ---
        # --- START MODIFICATION ---
        if intent in [UserIntent.IMAGE_GENERATION, UserIntent.SPEECH_PROCESSING, UserIntent.WEATHER_INQUIRY]:
            logger.info(f"Handling '{intent.value}' directly via specialist delegation.")
            
            specialist_id = None
            task_type = ""
            if intent == UserIntent.IMAGE_GENERATION:
                specialist_id = TENSORART_SPECIALIST_AGENT_ID
                task_type = "image"
            elif intent == UserIntent.SPEECH_PROCESSING:
                specialist_id = SPEECH_SPECIALIST_AGENT_ID
                task_type = "speech"
            elif intent == UserIntent.WEATHER_INQUIRY:
                specialist_id = WEATHER_SPECIALIST_AGENT_ID
                task_type = "weather"
            
            if not specialist_id:
                 return {"response": "Could not find a specialist for this task.", ...}

            # The rest of the delegation logic remains the same
            specialist_agent = self.agent_manager.get_agent_instance(specialist_id)
            # ...
        # --- END MODIFICATION ---

        # --- PATH B: CONVERSATIONAL TASK (The original NCF workflow) ---
        elif intent in [UserIntent.CONVERSATION, UserIntent.FILE_SEARCH]:
            # ... (existing NCF logic)
```

### Step 5: Create the Specialist Agent Instance

Think of the files you created in Steps 1-4 (`weather_tool.py`, `weather_specialist_instruction.py`, and the changes in `agent_manager.py`) as the **blueprint** for a Weather Specialist. You've defined what it *can do* and how it *should think*.

However, at this point, no actual agent instance based on that blueprint exists in your system's database or on its file system. The Orchestrator knows that if it sees a `WEATHER_INQUIRY` intent, it should look for an agent with the ID `d5e8a1b3-...`, but if it looks, it will find nothing.

**Step 5 is the act of "manufacturing" the agent from the blueprint.** It's a one-time action that creates the necessary database record and configuration files, making the specialist a real, addressable entity that the Orchestrator can delegate tasks to.

---

### The "How": Two Concrete Methods to Create the Instance

Here are two practical ways to perform this one-time creation, with detailed examples.

#### Method 1: Using the Admin API Endpoint (Recommended)

This is the cleanest and most robust way to do it, especially for a running system. We will use the `/admin/create-specialist-agents` endpoint that was added to `api/routes.py`.

First, you need to add your new Weather Specialist to the list of specialists that this endpoint knows how to create.

**1. Update `api/routes.py`:**

Find the `create_all_specialist_agents` function and add your new weather agent to the `specialists_to_create` dictionary.

```python
# In api/routes.py

# ... other imports
from weather_specialist_instruction import WEATHER_SPECIALIST_INSTRUCTION # Add this import
from agent_manager import WEATHER_SPECIALIST_AGENT_ID # Add this import

# ...

@app.post("/admin/create-specialist-agents", tags=["Admin"], response_model=dict)
async def create_all_specialist_agents(current_user: dict = Depends(verify_token)):
    # ... (admin check logic)

    specialists_to_create = {
        "TensorArt": { ... },
        "Speech": { ... },
        "MCP": { ... },
        "A2A": { ... },
        # --- ADD YOUR NEW SPECIALIST HERE ---
        "Weather": {
            "id": WEATHER_SPECIALIST_AGENT_ID,
            "persona": "An AI expert in providing weather forecasts.",
            "instruction": WEATHER_SPECIALIST_INSTRUCTION,
            "roles": ["specialist", "weather_inquiry"] # A role to identify its tools
        }
        # --- END OF ADDITION ---
    }

    # ... (the rest of the function remains the same)
```

**2. Run the API Command:**

Once your server is running, you can create all the specialists (including your new one) by sending a `POST` request to that endpoint. You can use a tool like `curl` from your terminal.

First, you need to log in as an admin user (e.g., `xupeta`) to get an authentication token.

```bash
# Step 2a: Log in and get the token
TOKEN=$(curl -X POST "http://localhost:8000/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "xupeta", "password": "your_password"}' \
     | jq -r .access_token)

echo "Got Token: $TOKEN"

# Step 2b: Call the admin endpoint to create the specialists
curl -X POST "http://localhost:8000/admin/create-specialist-agents" \
     -H "Authorization: Bearer $TOKEN"
```

**What Happens:**
*   The API receives the request.
*   It iterates through the `specialists_to_create` dictionary.
*   It sees the new "Weather" entry.
*   It checks if an agent with `WEATHER_SPECIALIST_AGENT_ID` already exists. If not...
*   It creates the database record and the `d5e8a1b3-....json` configuration file on the file system, correctly setting its `roles` to `["specialist", "weather_inquiry"]`.

**Advantages of this method:**
*   **Idempotent:** You can run it multiple times, and it will only create agents that don't already exist.
*   **Controlled:** It's a formal API action, easy to manage and secure.
*   **No Code Changes (after initial setup):** You don't need to write a separate script.

#### Method 2: Using a Startup Script (Good for Development)

This method involves writing a simple Python script that you run once to provision the specialist agents. This is useful during initial development or for automated setup routines.

**1. Create a `provision_specialists.py` script:**

Create a new file in your project's root directory.

```python
# In a new file: provision_specialists.py

import os
from datetime import datetime
from agent_manager import AgentManager, AgentConfig
from database.models import AgentRepository

# --- Import all specialist constants and instructions ---
from agent_manager import (
    TENSORART_SPECIALIST_AGENT_ID,
    SPEECH_SPECIALIST_AGENT_ID,
    WEATHER_SPECIALIST_AGENT_ID # Your new ID
)
from tensorart_specialist_instruction import TENSORART_SPECIALIST_INSTRUCTION
from speech_specialist_instruction import SPEECH_SPECIALIST_INSTRUCTION
from weather_specialist_instruction import WEATHER_SPECIALIST_INSTRUCTION # Your new instruction

# --- Configuration ---
# This should be the user_id of the system administrator
ADMIN_USER_ID = "40c8d42f-858d-4060-b5be-8cef6480e9a3" # Example: xupeta's ID

def provision():
    """Creates system specialist agents if they don't already exist."""
    print("🚀 Starting Specialist Agent Provisioning...")
    
    db_repo = AgentRepository()
    agent_manager = AgentManager(db_repo=db_repo)

    specialists = {
        "TensorArt": {
            "id": TENSORART_SPECIALIST_AGENT_ID,
            "persona": "Image generation expert.",
            "instruction": TENSORART_SPECIALIST_INSTRUCTION,
            "roles": ["specialist", "image_generation"]
        },
        "Speech": {
            "id": SPEECH_SPECIALIST_AGENT_ID,
            "persona": "Text-to-speech expert.",
            "instruction": SPEECH_SPECIALIST_INSTRUCTION,
            "roles": ["specialist", "speech_processing"]
        },
        "Weather": {
            "id": WEATHER_SPECIALIST_AGENT_ID,
            "persona": "Weather forecast expert.",
            "instruction": WEATHER_SPECIALIST_INSTRUCTION,
            "roles": ["specialist", "weather_inquiry"]
        }
    }

    for name, details in specialists.items():
        agent_id = details["id"]
        
        # Check if the agent's config file already exists
        if agent_id in agent_manager._agent_configs:
            print(f"✅ Specialist '{name}' (ID: {agent_id}) already exists. Skipping.")
            continue
        
        print(f"🔧 Creating Specialist '{name}' (ID: {agent_id})...")
        
        # We manually create the config to ensure the fixed ID and roles are used
        agent_path = agent_manager.base_storage_path / ADMIN_USER_ID / agent_id
        agent_path.mkdir(parents=True, exist_ok=True)
        
        config = AgentConfig(
            agent_id=agent_id,
            user_id=ADMIN_USER_ID,
            name=f"{name} Specialist",
            persona=details["persona"],
            detailed_persona=details["instruction"],
            model="openrouter/openai/gpt-4o-mini", # A cheap, fast model is fine for specialists
            created_at=datetime.now(),
            settings={"system_type": "ncf", "roles": details["roles"]}
        )
        
        # Save the config file and update the in-memory cache
        agent_manager._save_agent_config(config)
        agent_manager._agent_configs[agent_id] = config
        
        # Create the corresponding database record
        db_repo.create_agent(
            agent_id=agent_id,
            user_id=ADMIN_USER_ID,
            name=f"{name} Specialist",
            persona=details["persona"],
            detailed_persona=details["instruction"],
            model=config.model,
            is_public=False # Specialists are internal system components
        )
        print(f"👍 Successfully created Specialist '{name}'.")

    print("\n🎉 Provisioning complete!")


if __name__ == "__main__":
    provision()
```

**2. Run the script:**

From your terminal (with the virtual environment activated), simply run the script once.

```bash
python provision_specialists.py
```

---

### Summary of What Step 5 Accomplishes

Regardless of which method you choose, the end result is the same:

1.  **Database Record:** A new row is created in the `agents` table with the specialist's fixed UUID, its name, persona, and owner (the admin user).
2.  **Filesystem State:** A new configuration file is created on disk (e.g., `agent_storage/<admin_user_id>/d5e8a1b3-....json`). This file contains the agent's settings, including the crucial `roles: ["specialist", "weather_inquiry"]` part.
3.  **System Awareness:** The `AgentManager` loads this new configuration into its memory.

Now, when a user asks, "What's the weather in Paris?", the system flows exactly as designed:
1. Orchestrator receives the message.
2. The intent router classifies it as `WEATHER_INQUIRY`.
3. The `process_message` logic looks up the `WEATHER_SPECIALIST_AGENT_ID`.
4. The `AgentManager` finds the now-existing agent instance.
5. The message is delegated to the Weather Specialist, which uses its `WeatherClient` tool to get the answer and respond.
