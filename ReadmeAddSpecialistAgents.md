
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

You'll need to create the specialist agent in the system one time, similar to the TensorArt specialist. You can do this via an admin API endpoint or a startup script that calls the `agent_manager.create_agent` method using the hardcoded `WEATHER_SPECIALIST_AGENT_ID`.

That's it! You have now successfully extended the AuraCode framework with a new, fully integrated Specialist Agent. The Orchestrator will now automatically route all weather-related queries to your expert, which will use its dedicated tools and instructions to provide an answer. This pattern can be repeated for any number of specialized tasks.
