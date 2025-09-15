# tensorart_tool.py

import requests
import os
import json
import uuid
from typing import List, Dict, Any, Optional


class TensorArtClient:
    """
    Um cliente para interagir com a API da Tensor.Art, tratando requisições,
    autenticação e o cálculo de custos.
    """

    def __init__(self, api_key: Optional[str] = None, endpoint: str = "https://ap-east-1.tensorart.cloud"):
        self.endpoint = endpoint
        # A documentação da Tensor.Art não especifica o header, mas 'X-API-KEY' ou 'Authorization' são comuns.
        # Vamos assumir 'Authorization: Bearer <key>' que é um padrão moderno.
        # Se não funcionar, troque para 'X-API-KEY'.
        self.api_key = api_key or os.getenv("TENSORART_API_KEY")
        if not self.api_key:
            raise ValueError("TENSORART_API_KEY não foi encontrada nas variáveis de ambiente.")

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _make_request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Faz uma requisição genérica para a API e trata erros."""
        url = f"{self.endpoint}{path}"
        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_text = e.response.text
            print(f"Erro HTTP: {e.response.status_code} - {error_text}")
            return {"status": "error", "code": e.response.status_code, "message": error_text}
        except Exception as e:
            print(f"Erro na requisição: {e}")
            return {"status": "error", "code": 500, "message": str(e)}

    def submit_job(self, stages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submete um job de geração de imagem com uma lista de estágios."""
        payload = {
            "request_id": str(uuid.uuid4()),
            "stages": stages
        }
        return self._make_request("POST", "/v1/jobs", json=payload)

    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Verifica o status e o resultado de um job."""
        return self._make_request("GET", f"/v1/jobs/{job_id}")

    def calculate_credits(self, stages: List[Dict[str, Any]]) -> float:
        """
        Calcula o custo estimado em créditos para uma lista de estágios,
        baseado nas fórmulas da documentação.
        """
        total_credits = 0.0
        width, height, steps, count = 512, 512, 20, 1
        model_factor = 1.0

        # Encontra parâmetros iniciais que podem ser herdados
        for stage in stages:
            if stage.get("type") == "INPUT_INITIALIZE":
                count = stage.get("inputInitialize", {}).get("count", 1)
            if stage.get("type") == "DIFFUSION":
                d_params = stage.get("diffusion", {})
                width = d_params.get("width", width)
                height = d_params.get("height", height)
                steps = d_params.get("steps", steps)

        # Calcula o custo para cada estágio individualmente
        for stage in stages:
            stage_type = stage.get("type")

            if stage_type == "DIFFUSION":
                current_steps = stage.get("diffusion", {}).get("steps", steps)
                stage_credits = model_factor * count * (((current_steps + 4) // 5) / 5.0)
                total_credits += stage_credits
                # Atualiza os parâmetros para estágios futuros
                steps = current_steps
                width = stage.get("diffusion", {}).get("width", width)
                height = stage.get("diffusion", {}).get("height", height)

            elif stage_type in ["IMAGE_TO_ADETAILER", "IMAGE_TO_INPAINT", "IMAGE_TO_UPSCALER"]:
                s, w, h = steps, width, height  # Herda por padrão

                if stage_type == "IMAGE_TO_ADETAILER":
                    arg = stage.get("image_to_adetailer", {}).get("args", [{}])[0]
                    if arg.get("ad_use_steps") == "true":  # A API usa strings 'true'/'false'
                        s = steps
                    elif "adSteps" in arg:
                        s = int(arg["adSteps"])

                if stage_type == "IMAGE_TO_UPSCALER":
                    up_params = stage.get("imageToUpscaler", {})
                    s = up_params.get("hrSecondPassSteps", steps)
                    scale = up_params.get("hrScale", 1.0)
                    w = int(width * scale) if "hrScale" in up_params else up_params.get("hrResizeX", width)
                    h = int(height * scale) if "hrScale" in up_params else up_params.get("hrResizeY", height)

                term1 = model_factor * count * (((s + 4) // 5) / 5.0)
                # CEIL(x*2)/2
                pixels_in_millions = (w * h) / (1024.0 * 1024.0)
                term2 = (int(pixels_in_millions * 2 + 0.99999)) / 2.0
                stage_credits = term1 * term2
                total_credits += stage_credits

                # Atualiza width/height para o próximo estágio
                width, height = w, h

        return round(total_credits, 2)