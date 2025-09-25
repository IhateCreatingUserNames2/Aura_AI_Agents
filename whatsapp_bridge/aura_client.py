# whatsapp_bridge/aura_client.py
import httpx

AURA_API_BASE_URL = "http://localhost:8000" # Sua URL da API AURA

async def login(username, password):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{AURA_API_BASE_URL}/auth/login",
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            return response.json()
        return None

async def get_my_agents(token):
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AURA_API_BASE_URL}/agents/my-agents", headers=headers)
        if response.status_code == 200:
            return response.json()
        return []

async def chat_with_agent(token, agent_id, message):
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Usando o endpoint /chat/{agent_id} com método POST e FormData
        response = await client.post(f"{AURA_API_BASE_URL}/chat/{agent_id}", headers=headers, data=data)
        if response.status_code == 200:
            return response.json()
        return None