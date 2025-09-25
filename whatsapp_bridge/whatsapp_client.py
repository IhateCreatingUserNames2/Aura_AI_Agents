# whatsapp_bridge/whatsapp_client.py
import os
import logging
import httpx  # Usar httpx para chamadas assíncronas

WHATSAPP_TOKEN = os.getenv("WHATSAPP_PERMANENT_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERSION = "v23.0"

# Criamos um cliente assíncrono para reutilizar conexões (melhor performance)
async_client = httpx.AsyncClient(timeout=30.0)


async def send_whatsapp_message(to_number: str, message: str):
    """
    Envia mensagem via WhatsApp Business API de forma assíncrona e robusta.
    """
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        logging.error("❌ WHATSAPP_TOKEN ou PHONE_NUMBER_ID não configurados.")
        return None

    url = f"https://graph.facebook.com/{VERSION}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,  # A Meta espera o número limpo com código do país
        "type": "text",
        "text": {"body": message}
    }

    try:
        logging.info(f"📤 Enviando mensagem para {to_number}: {message[:50]}...")

        # A chamada agora é assíncrona, não bloqueia o servidor
        response = await async_client.post(url, headers=headers, json=data)

        logging.info(f"📨 Resposta da Meta - Status: {response.status_code}, Corpo: {response.text}")

        # Lança uma exceção para erros (4xx, 5xx), que será capturada abaixo
        response.raise_for_status()

        logging.info(f"✅ Mensagem para {to_number} aceita pela Meta.")
        return response.json()

    except httpx.HTTPStatusError as e:
        logging.error(f"❌ Erro HTTP ao enviar mensagem para {to_number}: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        logging.error(f"❌ Erro inesperado ao enviar mensagem: {e}", exc_info=True)
        return None