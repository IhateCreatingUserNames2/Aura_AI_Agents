# speech_tool.py
import requests
import os
from typing import Dict, Any, List

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# A curated list of models you want to offer
SUPPORTED_TTS_MODELS = {
    "espnet_female": "espnet/kan-bayashi_ljspeech_vits",
    "speecht5_male": "microsoft/speecht5_tts",
    "bark_versatile": "suno/bark-small",
}
SUPPORTED_STT_MODELS = {
    "whisper_large": "openai/whisper-large-v3",
    "distil_whisper": "distil-whisper/distil-large-v2",
}


class SpeechClient:
    def _query_hf(self, payload: Dict, model_id: str) -> bytes:
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        response = requests.post(api_url, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.content

    def list_available_models(self) -> Dict[str, List[str]]:
        """Lists the available models for Text-to-Speech and Speech-to-Text."""
        return {
            "tts_models": list(SUPPORTED_TTS_MODELS.keys()),
            "stt_models": list(SUPPORTED_STT_MODELS.keys()),
        }

    def generate_speech(self, text: str, model_key: str = "espnet_female") -> Dict[str, Any]:
        """
        Converts text to speech using a specified model.
        Returns a dictionary with a URL to the generated audio file.
        """
        model_id = SUPPORTED_TTS_MODELS.get(model_key)
        if not model_id:
            return {"status": "error", "message": f"Invalid model key. Use one of {list(SUPPORTED_TTS_MODELS.keys())}"}

        try:
            audio_bytes = self._query_hf({"inputs": text}, model_id)
            # In a real app, you'd save this to S3/local storage and return a URL
            # For now, we'll simulate this.
            # filename = f"speech_{uuid.uuid4()}.flac"
            # with open(f"agent_storage/_shared/audio/{filename}", "wb") as f:
            #     f.write(audio_bytes)
            # return {"status": "success", "audio_url": f"/shared/audio/{filename}"}
            return {"status": "success", "message": "Audio generated successfully (URL placeholder)."}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    # transcribe_audio would be similar, taking audio bytes and returning text.