# whatsapp_bridge/bridge_main.py
import os
import logging
from fastapi import FastAPI, Request, HTTPException, Response, BackgroundTasks
from dotenv import load_dotenv

from database import SessionLocal, WhatsAppUser
import aura_client
from whatsapp_client import send_whatsapp_message  # Agora importa a versão async

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI()

VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Handles the webhook verification challenge from Meta."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logging.info("✅ Webhook verificado com sucesso!")
        return Response(content=challenge, status_code=200)
    else:
        logging.error("❌ Falha na verificação do Webhook.")
        raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handles incoming messages from WhatsApp."""
    try:
        data = await request.json()
        logging.info(f"📦 Webhook recebido: {data}")

        message_info = extract_message_info(data)
        if message_info:
            phone_number, message_body = message_info
            logging.info(f"▶️ Mensagem válida de {phone_number} extraída. Processando em background.")
            background_tasks.add_task(process_incoming_message, phone_number, message_body)
        else:
            logging.info("ℹ️ Webhook recebido não é uma mensagem de texto de usuário. Ignorando.")

    except Exception as e:
        logging.error(f"❌ Erro crítico ao processar o corpo do webhook: {e}", exc_info=True)

    return Response(status_code=200)


def extract_message_info(data: dict):
    """Extrai informações da mensagem de forma segura e robusta."""
    try:
        if data.get("object") != "whatsapp_business_account":
            return None

        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                if change.get("field") == "messages":
                    value = change.get("value", {})
                    for message_data in value.get("messages", []):
                        if message_data.get("type") == "text":
                            phone_number = message_data.get("from")
                            message_body = message_data.get("text", {}).get("body")
                            if phone_number and message_body:
                                return phone_number, message_body
    except Exception as e:
        logging.error(f"🚨 Erro ao extrair info da mensagem: {e}", exc_info=True)

    return None


async def process_incoming_message(phone_number: str, message: str):
    """The main logic to handle user interaction state and commands."""
    logging.info(f"Processando mensagem de {phone_number}: {message}")
    db = SessionLocal()
    try:
        user = db.query(WhatsAppUser).filter(WhatsAppUser.phone_number == phone_number).first()

        if message.lower().startswith("!login"):
            await handle_login_command(phone_number, message, user, db)
            return
        if not user:
            await send_whatsapp_message(phone_number,
                                        "🌟 Bem-vindo ao AURA! Para começar, faça login com: !login <usuario> <senha>")
            return
        if message.lower() == "!agents":
            await handle_agents_command(phone_number, user)
            return
        if message.lower().startswith("!select"):
            await handle_select_command(phone_number, message, user, db)
            return
        if message.lower() == "!help":
            help_text = "🤖 *Comandos disponíveis:*\n\n!agents - Ver seus agentes\n!select <id> - Selecionar um agente\n!help - Mostrar esta ajuda\n\nApós selecionar um agente, envie qualquer mensagem para conversar!"
            await send_whatsapp_message(phone_number, help_text)
            return
        if not user.selected_agent_id:
            await send_whatsapp_message(phone_number,
                                        "❌ Nenhum agente selecionado. Use !agents para ver seus agentes e !select <id> para escolher um.")
            return
        await handle_chat_message(phone_number, message, user)
    except Exception as e:
        logging.error(f"Erro ao processar mensagem de {phone_number}: {e}", exc_info=True)
        await send_whatsapp_message(phone_number, "❌ Ocorreu um erro interno. Tente novamente.")
    finally:
        db.close()


async def handle_login_command(phone_number: str, message: str, user, db):
    parts = message.split()
    if len(parts) != 3:
        await send_whatsapp_message(phone_number, "❌ Comando inválido. Use: !login <usuario> <senha>")
        return
    _, username, password = parts
    auth_data = await aura_client.login(username, password)
    if auth_data:
        if user:
            user.aura_user_id = auth_data["user_id"]
            user.aura_auth_token = auth_data["access_token"]
        else:
            user = WhatsAppUser(phone_number=phone_number, aura_user_id=auth_data["user_id"],
                                aura_auth_token=auth_data["access_token"])
            db.add(user)
        db.commit()
        await send_whatsapp_message(phone_number,
                                    "✅ Login realizado com sucesso! Use !agents para ver seus agentes ou !select <id> para começar a conversar.")
    else:
        await send_whatsapp_message(phone_number, "❌ Login falhou. Verifique seu usuário e senha.")


async def handle_agents_command(phone_number: str, user):
    agents = await aura_client.get_my_agents(user.aura_auth_token)
    if agents:
        response_text = "🤖 *Seus Agentes:*\n\n"
        for agent in agents:
            response_text += f"• {agent['name']} (ID: `{agent['agent_id']}`)\n"
        response_text += "\n💬 Use !select <id> para conversar com um agente"
        await send_whatsapp_message(phone_number, response_text)
    else:
        await send_whatsapp_message(phone_number, "❌ Você ainda não criou nenhum agente.")


async def handle_select_command(phone_number: str, message: str, user, db):
    parts = message.split()
    if len(parts) != 2:
        await send_whatsapp_message(phone_number, "❌ Comando inválido. Use: !select <id_agente>")
        return
    _, agent_id = parts
    agents = await aura_client.get_my_agents(user.aura_auth_token)
    if agents and any(agent['agent_id'] == agent_id for agent in agents):
        user.selected_agent_id = agent_id
        db.commit()
        await send_whatsapp_message(phone_number, f"✅ Agora você está conversando com o agente `{agent_id}`")
    else:
        await send_whatsapp_message(phone_number,
                                    f"❌ Agente `{agent_id}` não encontrado. Use !agents para ver seus agentes.")


async def handle_chat_message(phone_number: str, message: str, user):
    aura_response = await aura_client.chat_with_agent(user.aura_auth_token, user.selected_agent_id, message)
    if aura_response and "response" in aura_response:
        await send_whatsapp_message(phone_number, aura_response["response"])
    else:
        await send_whatsapp_message(phone_number, "❌ Desculpe, houve um erro ao se comunicar com o agente.")


if __name__ == "__main__":
    import uvicorn

    required_env = ["WHATSAPP_PERMANENT_TOKEN", "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_VERIFY_TOKEN"]
    if any(not os.getenv(env) for env in required_env):
        logging.error(f"❌ Variáveis de ambiente faltando: {[env for env in required_env if not os.getenv(env)]}")
        exit(1)
    logging.info("🚀 Iniciando WhatsApp Bridge...")
    uvicorn.run(app, host="0.0.0.0", port=8001)