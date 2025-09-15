# A Technical Guide to CEAF's Precision Mode (DeepConf)

This document provides a technical explanation of the **Precision Mode** within the **Coherent Emergence Agent Framework (CEAF)**. This mode is an advanced reasoning mechanism designed for situations that demand high accuracy, deliberation, and confidence. It is a direct implementation of the principles described in the research paper on **Deep Confidence-based early-exiting (DeepConf)**.

## 🎯 What is Precision Mode?

Precision Mode is a special operational state for a CEAF agent that fundamentally changes how it "thinks." Instead of generating a single response in a straightforward manner, the agent initiates a complex, multi-path reasoning process to arrive at a more robust and confident conclusion.

-   **Standard (Fast) Mode:** 1 thought process → 1 answer.
-   **Precision (DeepConf) Mode:** Multiple parallel thought processes → Consensus → 1 confident answer.

This mode is computationally expensive and significantly slower, but it dramatically reduces the likelihood of hallucination and improves the quality of reasoning on complex tasks.

### When to Use It

-   When the user's query is complex, ambiguous, or has high stakes.
-   When the agent needs to perform multi-step reasoning.
-   When the user explicitly requests higher accuracy (e.g., "be more precise," "double-check that," "think carefully").

The agent's **Metacognitive Control Loop (MCL)** can also autonomously decide to increase its reasoning budget if it enters a state of `FAILING_PRODUCTIVELY`, mimicking a human's "stop and think" response when confused.

## ⚙️ The Core Algorithm: DeepConf

Precision Mode is powered by the **DeepConf** algorithm, which is orchestrated by the `ORA` (Orchestrator/Responder Agent) module. The goal is to simulate a "committee of experts" in the agent's mind and only accept an answer that the committee agrees on.

This is implemented in the `_generate_response()` and `_streaming_call_with_deepconf()` methods within `ceaf_system/ORA.py`.

### Step 1: Measuring Confidence with Logprobes

The entire algorithm depends on a way to measure the LLM's confidence at each step of its generation. We do this by analyzing **log probabilities (logprobs)**, which are requested from the OpenRouter API during a streaming response.

For each token the LLM generates (e.g., the word " apple"), the API also tells us the log probability of that token, *and* the log probabilities of other likely tokens (e.g., " pear," " orange").

The **Logprob Analyzer** (`ceaf_system/logprob_analyzer.py`) calculates a confidence score for each token using a specific formula from the DeepConf paper:

> **Confidence = - (Average Logprob of Top-k Candidate Tokens)**

A higher score means the LLM was more "certain" about its choice because the alternative tokens had much lower probabilities. A lower score means the LLM was uncertain, as several other tokens were nearly as likely.

### Step 2: The DeepConf Workflow

When Precision Mode is active, the ORA executes the following steps:

#### **Phase 1: Warmup & Dynamic Thresholding**

1.  **Parallel Traces:** The ORA initiates a "warmup" phase by running multiple (e.g., 3) LLM generation processes in parallel. Each process is called a "trace" or a "thought path."
2.  **Confidence Monitoring:** For each trace, it calculates the confidence score for every token generated, creating a stream of confidence values.
3.  **Sliding Window Analysis:** It uses a sliding window (e.g., 20 tokens) to find the point of *minimum average confidence* in each trace. This identifies the weakest or most uncertain part of the reasoning path.
4.  **Dynamic Threshold Calculation:** After the warmup traces are complete, the system takes all the minimum confidence scores and calculates a percentile (e.g., the 10th percentile). This becomes the **dynamic confidence threshold** for the next phase. This step is crucial: the agent learns the "difficulty" of the current problem and sets a reasonable confidence bar for itself.

#### **Phase 2: Final Generation with Early Exiting**

1.  **More Parallel Traces:** The ORA launches the remaining budget of parallel traces (e.g., 5-9 more).
2.  **Real-time Vetoing:** As these new traces generate text, their confidence is monitored in real-time. If the average confidence within the sliding window drops **below the dynamic threshold** calculated in Phase 1, that trace is immediately terminated ("vetoed"). This is the "early exiting" mechanism. It saves computation by discarding unpromising lines of thought as soon as they become incoherent.
3.  **Trace Completion:** Traces that successfully complete generation without being vetoed are considered "valid" reasoning paths.

#### **Phase 3: Wisdom-based Voting & Final Answer Selection**

1.  **Gather Candidates:** All valid traces from both phases are collected. If no traces survived, the agent reports that it could not form a confident answer.
2.  **Calculate Wisdom Score:** If there are valid traces, the system doesn't just pick one at random. It calculates a **Wisdom Score** for each valid trace. This score is a weighted average of:
    -   **Narrative Coherence (from NCIM):** How well does this thought path align with the agent's core identity and past experiences? (High weight)
    -   **Minimum Confidence:** How confident was this thought path during its weakest moment? (Low weight)
3.  **Final Selection:** The trace with the highest Wisdom Score is chosen. The final, user-facing answer is extracted from this "wisest" line of thought.

This final step ensures that the agent's response is not only logically sound (high confidence) but also **true to its character** (high narrative coherence).

## 🔧 How to Modify Precision Mode

You can tune the behavior of Precision Mode by adjusting parameters in the `MCL` and `ORA` modules.

### In `ceaf_system/MCL.py` - `get_next_turn_parameters()`:

This method defines the "budget" for Precision Mode.

```python
# In ceaf_system/MCL.py

def get_next_turn_parameters(self, precision_mode: bool = False) -> Dict[str, Any]:
    if precision_mode:
        # PARAMETERS FOR PRECISION MODE (SLOW & ROBUST)
        logger.info("MCL: Generating parameters for PRECISION MODE.")
        params = {
            "temperature": 0.7,            # Adjust creativity
            "confidence_threshold": 16.8,  # Base threshold (can be overridden by dynamic one)
            "warmup_traces": 3,            # Increase/decrease warmup budget
            "total_budget": 8              # Total parallel thoughts to generate
        }
    # ...
```

-   **`warmup_traces`**: Increasing this gives the system a better sample for setting the dynamic threshold but costs more.
-   **`total_budget`**: This is the total number of parallel LLM calls. Higher values increase the chance of finding a valid answer but dramatically increase cost and latency.

### In `ceaf_system/ORA.py` - `_streaming_call_with_deepconf()`:

This method contains the sliding window size.

```python
# In ceaf_system/ORA.py

async def _streaming_call_with_deepconf(...):
    # ...
    WINDOW_SIZE = mcl_params.get("confidence_window", 2048) # Default is 2048 characters, not tokens
    # In the code, it's actually per token, so let's use a more reasonable default
    WINDOW_SIZE = 20 # A 20-token window is more realistic
    # ...
```

-   **`WINDOW_SIZE`**: A larger window makes the confidence score smoother and less sensitive to single-token fluctuations. A smaller window makes it more reactive and quicker to veto unstable traces.

## 📈 Summary

Precision Mode is a powerful but resource-intensive feature. It transforms the CEAF agent from a simple response generator into a deliberative reasoning engine that actively manages its own uncertainty. By simulating and filtering multiple lines of thought, it produces answers that are more reliable, coherent, and aligned with the agent's core identity.
