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
