# ==================== api/routes.py
"""
Enhanced API routes with memory management, agent editing, and pre-built agent capabilities
"""

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import litellm as Litellmtop
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any, Union
import jwt
import re
from datetime import datetime, timedelta, timezone
import bcrypt
import os
from pathlib import Path
import sys
import logging
import json
import io
import zipfile
import asyncio
import uuid
import shutil
from rag_processor import process_and_index_file
from tensorart_specialist_instruction import TENSORART_SPECIALIST_INSTRUCTION
from agent_manager import TENSORART_SPECIALIST_AGENT_ID
from ncf_processing import montar_prompt_aura_ncf

# Import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .meme_selector_routes import router as meme_selector_router

# Original imports
from agent_manager import AgentManager, AgentConfig, MODEL_COSTS, NCFAuraAgentInstance
from database.models import AgentRepository, User, Agent, Memory, CreditTransaction, ChatSession, Message
from memory_system.memory_models import Memory as MemoryModel
from sqlalchemy.orm import joinedload
from sqlalchemy import desc, and_
from ncf_processing import aura_reflector_analisar_interacao, analisar_e_contribuir_para_memoria_live
from langchain_community.chat_models import litellm
from ceaf_adapter import CEAFAgentAdapter

# Imports for Pre-built Agent Extensions
from prebuilt_agents_system import (
    PrebuiltAgentRepository,
    AgentArchetype,
    AgentMaturityLevel,
    PersonalityArchitect
)

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Initialize FastAPI app
app = FastAPI(
    title="Aura Multi-Agent API (NCF-Enabled)",
    description="API for creating and managing multiple NCF-powered Aura AI agents with advanced memory, context, and pre-built agent capabilities",
    version="2.1.0"
)

app.include_router(meme_selector_router)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CORRECTED INITIALIZATION WITH ABSOLUTE PATHS ---
# Determine project root dynamically.
# If routes.py is in 'api/', and 'agent_storage' & 'aura_agents.db' are at the project root level:
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_STORAGE_PATH = PROJECT_ROOT / "agent_storage"
DATABASE_FILE_PATH = PROJECT_ROOT / "aura_agents.db"
PREBUILT_AGENTS_PATH = PROJECT_ROOT / "prebuilt_agents"

# Ensure the agent storage directory exists
AGENT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
logger.info(f"Using AGENT_STORAGE_PATH: {AGENT_STORAGE_PATH.resolve()}")
logger.info(f"Using DATABASE_FILE_PATH: {DATABASE_FILE_PATH.resolve()}")

# --- ENHANCED SERVICES & PLACEHOLDERS ---

# --- SERVICES INITIALIZATION ---

db_repo = AgentRepository(db_url=f"sqlite:///{str(DATABASE_FILE_PATH)}")
agent_manager = AgentManager(base_storage_path=str(AGENT_STORAGE_PATH), db_repo=db_repo)
prebuilt_repo = PrebuiltAgentRepository(storage_path=str(PROJECT_ROOT / "prebuilt_agents"))

if not prebuilt_repo.agents:
    try:
        from prebuilt_agents_system import create_sample_prebuilt_agents

        prebuilt_repo = create_sample_prebuilt_agents()
        logger.info("Loaded sample pre-built agents.")
    except Exception as e:
        logger.error(f"Could not load sample pre-built agents: {e}")

security = HTTPBearer()


# --- Pydantic Models ---

# --- Novos Modelos Pydantic para Chat History ---
class SaveMessageRequest(BaseModel):
    session_id: str
    role: str  # 'user' or 'assistant'
    content: str


class UpdateLiveMemoryPermissionsRequest(BaseModel):
    allow_contribution: bool
    allow_influence: bool


class FileResponseModel(BaseModel):
    filename: str
    content_type: str
    size: int
    url: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    total_messages: int
    created_at: str
    last_active: str


class SessionResponse(BaseModel):
    session_id: str
    agent_id: str
    agent_name: str
    created_at: str
    last_active: str
    message_count: int


# --- ENDPOINTS PARA CHAT HISTORY ---

class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str


class IdentityHistoryEntry(BaseModel):
    timestamp: str
    narrative: str


class IdentityHistoryResponse(BaseModel):
    agent_id: str
    system_type: str
    history: List[IdentityHistoryEntry]


class UserLoginRequest(BaseModel):
    username: str
    password: str


class CreateAgentRequest(BaseModel):
    name: str
    persona: str
    detailed_persona: str
    model: Optional[str] = None
    system_type: str = Field("ncf", description="The type of agent system to use ('ncf' or 'ceaf').")
    settings: Optional[Dict[str, Any]] = None  # For CEAF settings


class UpdateAgentRequest(BaseModel):
    name: Optional[str] = None
    persona: Optional[str] = None
    detailed_persona: Optional[str] = None
    avatar_url: Optional[str] = None
    is_public: Optional[bool] = None
    model: Optional[str] = None
    settings: Optional[dict] = None


class UpdateModelRequest(BaseModel):
    model: str


class EnhancedUpdateAgentRequest(BaseModel):
    name: Optional[str] = None
    persona: Optional[str] = None
    detailed_persona: Optional[str] = None
    avatar_url: Optional[str] = None
    is_public: Optional[bool] = None
    settings: Optional[dict] = None
    model: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    session_overrides: Optional[Dict[str, Any]] = None # <-- ADD THIS


class AgentResponse(BaseModel):
    agent_id: str
    name: str
    persona: str
    detailed_persona: str
    created_at: str
    is_public: bool
    owner_username: Optional[str] = None
    capabilities: List[str]
    settings: Optional[Dict[str, Any]] = None
    avatar_url: Optional[str] = None
    version: Optional[str] = "1.0.0"
    clone_count: int = 0


class ChatResponse(BaseModel):
    response: str
    session_id: str
    ncf_enabled: bool = True
    timestamp: Optional[str] = None


class AddCreditsRequest(BaseModel):
    amount: int
    description: str


# ==============================================================================
# ==================== NEW MODELS FOR BIOGRAPHY CREATION =======================
# ==============================================================================

class AgentBiography(BaseModel):
    class BiographyConfig(BaseModel):
        name: str
        persona: str
        detailed_persona: str
        system_type: str = "ncf"
        model: str = "openrouter/openai/gpt-4o-mini"
        is_public: bool = False

    class BiographyMemory(BaseModel):
        content: str
        memory_type: str
        emotion_score: float = 0.0
        initial_salience: float = 0.5
        custom_metadata: Dict[str, Any] = Field(default_factory=dict)

    config: BiographyConfig
    biography: List[BiographyMemory]


# ==============================================================================
# ================================= END OF NEW MODELS ==========================
# ==============================================================================

class MemorySearchRequest(BaseModel):
    query: str
    memory_types: Optional[List[str]] = None
    limit: Optional[int] = 10


class MemoryUploadRequest(BaseModel):
    memories: List[Dict[str, Any]]
    overwrite_existing: Optional[bool] = False
    validate_format: Optional[bool] = True


class MyAgentResponse(AgentResponse):
    is_public_template: bool
    version: Optional[str] = None
    clone_count: int = 0
    usage_cost: float = 0.0
    model: Optional[str] = None


class SetPriceRequest(BaseModel):
    usage_cost: float = Field(..., ge=0, description="The cost in credits to use this agent.")


class AddBiographyRequest(BaseModel):
    biography: List[AgentBiography.BiographyMemory]


class MemoryExportResponse(BaseModel):
    agent_id: str
    agent_name: str
    export_timestamp: str
    total_memories: int
    memory_types: List[str]
    memories: List[Dict[str, Any]]


class BulkMemoryUploadResponse(BaseModel):
    total_uploaded: int
    successful_uploads: int
    failed_uploads: int
    errors: List[str]
    memory_types_added: List[str]


class PrebuiltAgentResponse(BaseModel):
    id: str
    name: str
    archetype: str
    maturity_level: str
    system_type: str
    short_description: str
    # ADDED THIS FIELD:
    detailed_persona: Optional[str] = None
    total_interactions: int
    rating: float
    download_count: int
    tags: List[str]
    sample_memories: List[dict] = Field(default_factory=list)
    # ADDED THESE FIELDS:
    breakthrough_count: Optional[int] = None
    coherence_average: Optional[float] = None
    created_by: Optional[str] = "Cognai"


class CloneAgentRequest(BaseModel):
    source_agent_id: str
    custom_name: Optional[str] = None
    clone_memories: bool = True


class CreatePrebuiltRequest(BaseModel):
    """Request to create a new pre-built agent (admin only)"""
    name: str
    archetype: str
    system_type: str
    custom_traits: Optional[List[str]] = None
    initial_conversations: Optional[List[dict]] = None


class TrainingConversation(BaseModel):
    """Conversation for personality training"""
    user_message: str
    agent_response: str
    rating: float  # 1-5
    notes: Optional[str] = None


class TrainAgentRequest(BaseModel):
    """Request to train a pre-built agent"""
    agent_id: str
    conversations: List[TrainingConversation]
    target_traits: Optional[List[str]] = None


# --- Authentication Helpers ---
def create_access_token(user_id: str, username: str) -> str:
    """Create JWT access token"""
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode = {
        "user_id": user_id,
        "username": username,
        "exp": expire
    }
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def sanitize_agent_name(name: str) -> str:
    """
    Sanitizes a string to be a valid identifier for LlmAgent.
    - Replaces spaces and invalid characters with underscores.
    - Ensures it starts with a letter or underscore.
    - Removes consecutive underscores.
    """
    # Replace spaces and other problematic characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    # Remove leading characters that are not letters or underscore
    name = re.sub(r'^[^a-zA-Z_]+', '', name)

    # Ensure it doesn't start with a number if it's the only thing left
    if name and name[0].isdigit():
        name = '_' + name

    # Replace multiple underscores with a single one
    name = re.sub(r'_+', '_', name)

    # If the name is empty after sanitization, provide a default
    if not name:
        return f"agent_{uuid.uuid4().hex[:8]}"

    return name


async def initiate_agent_date(agent_a_id: str, agent_b_id: str, agent_manager):
    """Orchestrates a 'date' between two agents."""
    agent_a = agent_manager.get_agent_instance(agent_a_id)
    agent_b = agent_manager.get_agent_instance(agent_b_id)

    # 1. Initial Probing Question from Agent A
    initial_prompt = f"Hello, I am {agent_a.config.name}. I am here to learn about you. Based on your core identity, what is a topic you are passionate about?"

    # Use the A2A wrapper or a direct process_message call for simulation
    response_b = await agent_b.process_message(
        user_id="dating_simulation",
        session_id=f"date_{agent_a_id}_{agent_b_id}",
        message=initial_prompt
    )

    # 2. Agent A Analyzes the Response
    analysis_prompt_a = f"Analyze this response from {agent_b.config.name}: '{response_b['response']}'. What does this tell you about their personality? Is it compatible with my user's traits?"

    analysis_a = await agent_a.process_message(
        user_id="dating_simulation",
        session_id=f"date_{agent_a_id}_{agent_b_id}",
        message=analysis_prompt_a
    )

    # 3. Generate a Match Report (This is the key output)
    report = generate_match_report(agent_a, agent_b, response_b['response'], analysis_a['response'])

    return report


def generate_match_report(agent_a, agent_b, initial_response, analysis):
    # Use Agent A's LLM to write a summary for its user
    # This is a simplified example
    summary = f"I had a conversation with {agent_b.config.name}'s AI. When I asked about its passions, it said: '{initial_response}'. My analysis suggests: '{analysis}'. Based on this, I believe there is a potential for a good connection."

    # Calculate a score (e.g., based on emotion scores of memories created during the date)
    match_score = 75.0  # Placeholder

    return {"match_agent_id": agent_b.config.agent_id, "summary": summary, "score": match_score}


async def run_post_chat_analysis(
        user_utterance: str,
        agent_response: str,
        ncf_prompt: str,
        domain_context: str,
        agent_id: str,
        agent_model_name: str
):
    """
    Função de trabalho em segundo plano que executa todas as análises pós-chat.
    CORRIGIDO: Agora lida corretamente com agentes NCF e CEAF.
    """
    logger.info(f"Background worker: Iniciando análise pós-chat para o agente {agent_id}.")

    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance:
        logger.error(f"Background worker: Não foi possível encontrar a instância do agente {agent_id} para análise.")
        return

    # A lógica para determinar llm_for_analysis continua a mesma
    llm_for_analysis = None
    if hasattr(agent_instance, 'model'):
        llm_for_analysis = agent_instance.model
    elif hasattr(agent_instance, 'config') and hasattr(agent_instance.config, 'model'):
        model_name_config = agent_instance.config.model
        llm_for_analysis = LiteLlm(model=model_name_config)
    else:
        logger.error(f"Background worker: Não foi possível determinar a instância LLM para o agente {agent_id}.")
        return

    # Etapa A: Reflexão Privada
    await aura_reflector_analisar_interacao(
        agent_config=agent_instance.config,
        db_repo=db_repo,
        user_utterance=user_utterance,
        prompt_ncf_usado=ncf_prompt,
        resposta_de_aura=agent_response,
        memory_blossom=agent_instance.memory_blossom,
        user_id=agent_instance.config.user_id,
        llm_instance=llm_for_analysis,
        domain_context=domain_context,
        agent_model_name=agent_model_name  # PASSAR O PARÂMETRO ADIANTE
    )

    # Etapa B: Análise para Contribuição Pública
    await analisar_e_contribuir_para_memoria_live(
        user_utterance=user_utterance,
        resposta_de_aura=agent_response,
        agent_config=agent_instance.config,
        db_repo=db_repo
    )
    logger.info(f"Background worker: Análise pós-chat para o agente {agent_id} concluída.")


# --- FUNÇÃO AUXILIAR COMUM ---
async def process_chat_message(
        agent_id: str,
        message: str,
        session_id: Optional[str],
        current_user: dict,
        background_tasks: BackgroundTasks,
        save_to_history: bool = True,
        session_overrides: Optional[Dict[str, Any]] = None
) -> dict:
    """
    Função comum para processar mensagens de chat com persistência opcional
    COMPATÍVEL COM NCF E CEAF
    """
    max_input_words = 8046
    if len(message.split()) > max_input_words:
        raise HTTPException(
            status_code=413,
            detail=f"Your message is too long. Please keep it under {max_input_words} words."
        )

    try:
        # 1. Obter instância do agente (NCF ou CEAF via adapter)
        agent = agent_manager.get_agent_instance(agent_id)
        if not agent:
            agent = agent_manager.get_agent_instance(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")

        # 2. Verificar permissões
        config = agent_manager._agent_configs.get(agent_id)
        if not config:
            raise HTTPException(status_code=404, detail="Agent configuration not found")

        # ======================= START OF FIX =======================
        # This is the corrected permission logic block.
        if config.user_id != current_user["user_id"]:
            # If the user is not the owner, we must check if it's a public template.
            db_agent = db_repo.get_agent(agent_id)
            # The check is now against `is_public_template`.
            if not db_agent or not db_agent.is_public_template:
                raise HTTPException(status_code=403, detail="Access denied")
        # ======================== END OF FIX ========================

        model_for_response_and_analysis = config.model
        if session_overrides and 'model' in session_overrides:
            model_for_response_and_analysis = session_overrides['model']

        # 3. Gerenciar sessão se necessário
        chat_session_id = None
        if save_to_history:
            with db_repo.SessionLocal() as db_session:
                if session_id:
                    chat_session = db_session.query(ChatSession).filter(
                        and_(
                            ChatSession.id == session_id,
                            ChatSession.user_id == current_user["user_id"]
                        )
                    ).first()
                    if chat_session:
                        chat_session_id = chat_session.id
                        chat_session.last_active = datetime.utcnow()
                        db_session.commit()

                if not chat_session_id:
                    new_session = ChatSession(
                        user_id=current_user["user_id"],
                        agent_id=agent_id,
                        created_at=datetime.utcnow(),
                        last_active=datetime.utcnow()
                    )
                    db_session.add(new_session)
                    db_session.commit()
                    db_session.refresh(new_session)
                    chat_session_id = new_session.id
                    session_id = new_session.id
        else:
            if not session_id:
                session_id = f"temp_{current_user['user_id']}_{int(datetime.utcnow().timestamp())}"

        # 4. Salvar mensagem do usuário (se persistência habilitada)
        if save_to_history and chat_session_id:
            with db_repo.SessionLocal() as db_session:
                user_message = Message(
                    session_id=chat_session_id,
                    role="user",
                    content=message,
                    created_at=datetime.utcnow()
                )
                db_session.add(user_message)
                db_session.commit()

        response = ""
        system_type = agent_manager.get_agent_system_type(agent_id)

        try:
            # This logic block for processing NCF vs CEAF remains the same.
            if isinstance(agent, NCFAuraAgentInstance):
                result_dict = await agent.process_message(
                    user_id=current_user["user_id"],
                    session_id=session_id or "temp",
                    message=message,
                    session_overrides=session_overrides
                )
                response = result_dict["response"]
                logger.info(f"API: Scheduling background task for NCF reflector analysis.")
                background_tasks.add_task(
                    run_post_chat_analysis,
                    user_utterance=message,
                    agent_response=response,
                    ncf_prompt=result_dict["prompt"],
                    domain_context=result_dict["domain_context"],
                    agent_id=agent_id,
                    agent_model_name=model_for_response_and_analysis
                )
            else:  # CEAF Agent
                response_dict = await agent.process_message(
                    user_id=current_user["user_id"],
                    session_id=session_id or "temp",
                    message=message,
                    session_overrides=session_overrides
                )
                response = response_dict if isinstance(response_dict, str) else response_dict.get("response", "Error in response.")
                logger.info(f"API: Scheduling background task for CEAF post-chat analysis.")
                background_tasks.add_task(
                    run_post_chat_analysis,
                    user_utterance=message,
                    agent_response=response,
                    ncf_prompt=f"CEAF Interaction Context...",
                    domain_context="general",
                    agent_id=agent_id,
                    agent_model_name=model_for_response_and_analysis
                )
        except Exception as e:
            logger.error(f"Error processing message with agent: {e}", exc_info=True)
            response = "Sorry, there was an error processing your message. Please try again."
            system_type = "unknown"

        # 6. Salvar resposta do agente (se persistência habilitada)
        if save_to_history and chat_session_id:
            with db_repo.SessionLocal() as db_session:
                agent_message = Message(
                    session_id=chat_session_id,
                    role="assistant",
                    content=response,
                    created_at=datetime.utcnow()
                )
                db_session.add(agent_message)
                db_session.commit()

        return {
            "response": response,
            "session_id": session_id,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "ncf_enabled": True,
            "system_type": system_type,
            "agent_name": config.name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in process_chat_message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat processing failed")


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token and return user info"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {
            "user_id": payload["user_id"],
            "username": payload["username"]
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    # ==================== INÍCIO DA CORREÇÃO ====================
    # A exceção correta na biblioteca PyJWT é PyJWTError ou exceções mais específicas.
    # Vamos capturar a exceção específica e a geral para robustez.
    except (jwt.InvalidTokenError, jwt.PyJWTError) as e:
        logger.warning(f"Invalid JWT Token received: {e}")  # Adiciona um log para depuração
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )
    # ===================== FIM DA CORREÇÃO ======================


@app.post("/dating/find-matches/{agent_id}", tags=["AI Dating"])
async def find_ai_matches(agent_id: str, current_user: dict = Depends(verify_token)):
    """
    Instructs the user's agent to 'date' other public agents and find potential matches.
    """
    my_agent_config = agent_manager._agent_configs.get(agent_id)
    if not my_agent_config or my_agent_config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")

    # Get a list of public agents to "date" (excluding the user's own)
    public_agents = [
        agent for agent in agent_manager._agent_configs.values()
        if db_repo.get_agent(agent.agent_id).is_public_template and agent.agent_id != agent_id
    ]

    match_reports = []
    # In production, this should be a background task
    for target_agent in public_agents[:5]:  # Limit to 5 for now
        report = await initiate_agent_date(agent_id, target_agent.agent_id, agent_manager)
        match_reports.append(report)

    # Sort by score
    sorted_matches = sorted(match_reports, key=lambda x: x['score'], reverse=True)

    return {"matches": sorted_matches}

# --- User Authentication Routes ---
@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register_user(request: UserRegisterRequest):
    """Register a new user"""
    # Check if user exists
    with db_repo.SessionLocal() as session:
        existing_user = session.query(User).filter(
            (User.email == request.email) | (User.username == request.username)
        ).first()

        if existing_user:
            if existing_user.email == request.email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken"
                )

        # Hash password
        password_hash = bcrypt.hashpw(request.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Create user
        new_user = User(
            email=request.email,
            username=request.username,
            password_hash=password_hash
        )

        session.add(new_user)
        session.commit()
        session.refresh(new_user)

        # Create access token
        access_token = create_access_token(new_user.id, new_user.username)

        return {
            "user_id": new_user.id,
            "username": new_user.username,
            "email": new_user.email,
            "credits": new_user.credits,
            "access_token": access_token,
            "token_type": "bearer",
            "message": "Welcome! You can now create NCF-powered Aura agents with advanced memory and contextual understanding."
        }


@app.post("/auth/login")
async def login_user(request: UserLoginRequest):
    """Login user and return access token"""
    with db_repo.SessionLocal() as session:
        user = session.query(User).filter(User.username == request.username).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Verify password
        if not bcrypt.checkpw(request.password.encode('utf-8'), user.password_hash.encode('utf-8')):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Create access token
        access_token = create_access_token(user.id, user.username)

        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "access_token": access_token,
            "token_type": "bearer"
        }



# Define the path to the new frontend directory
memeselector_frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "memeselector"

# Mount the entire directory to the /memeselector path.
# The `html=True` option tells FastAPI to automatically serve `index.html`
# for requests to the root of the mounted path (i.e., "/memeselector").
if memeselector_frontend_path.is_dir():
    app.mount("/memeselector", StaticFiles(directory=memeselector_frontend_path, html=True), name="memeselector-frontend")
    print(f"INFO: Mounted Meme Selector frontend at /memeselector from {memeselector_frontend_path}")
else:
    print(f"WARNING: Meme Selector frontend directory not found at {memeselector_frontend_path}. The /memeselector URL will not work.")

@app.get("/auth/me")
async def get_current_user(current_user: dict = Depends(verify_token)):
    """Get current user information"""
    with db_repo.SessionLocal() as session:
        user = session.query(User).filter(User.id == current_user["user_id"]).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "credits": user.credits,
            "created_at": user.created_at.isoformat(),
            "agent_capabilities": ["ncf", "memory", "narrative_foundation", "rag", "reflector"]
        }


# --- Model Selector Routes ---
@app.get("/models/openrouter", tags=["Models"])
async def get_openrouter_models():
    """Provides a curated and tiered list of models with their credit costs."""
    # This list is structured to guide users from low-cost to high-value models.
    recommended_models = {
        "Free & Beta (Try Me!)": [
            "openrouter/deepseek/deepseek-chat-v3-0324:free",
            "openrouter/deepseek/deepseek-r1-0528:free",
            "openrouter/horizon-beta",
        ],
        "Featured (Best Value)": [
            "openrouter/openai/gpt-4o-mini",
            "openrouter/openai/gpt-4.1-mini",
            "openrouter/google/gemini-2.5-flash",
            "openrouter/mistralai/mistral-nemo",
        ],
        "Advanced (Maximum Power)": [
            "openrouter/openai/gpt-4o",
            "openrouter/anthropic/claude-sonnet-4",
            "openrouter/openai/gpt-4.1",
            "openrouter/google/gemini-2.5-pro",
            "openrouter/x-ai/grok-4",
        ],
        "Specialized & Niche": [
            "openrouter/qwen/qwen3-30b-a3b-instruct-2507",
            "openrouter/deepseek/deepseek-chat-v3-0324",
            "openrouter/anthropic/claude-3.7-sonnet",
            "openrouter/deepseek/deepseek-r1-0528",
        ]
    }

    # Create a new dictionary to hold models with their costs
    models_with_costs = {}

    for category, model_list in recommended_models.items():
        category_list = []
        for model_name in model_list:
            # Look up the cost, defaulting if not found
            cost = MODEL_COSTS.get(model_name, MODEL_COSTS["default"])
            category_list.append({"name": model_name, "cost": cost})
        models_with_costs[category] = category_list

    return models_with_costs


@app.get("/agents/my-agents", response_model=List[MyAgentResponse], tags=["Agent Management"])
async def get_my_agents(current_user: dict = Depends(verify_token)):
    """Fetches all agents (private and public templates) owned by the current user."""
    with db_repo.SessionLocal() as session:
        my_agents_db = session.query(Agent).filter(
            Agent.user_id == current_user["user_id"]
        ).options(
            joinedload(Agent.user)
        ).order_by(desc(Agent.created_at)).all()

    response_list = []
    for agent_db in my_agents_db:
        settings = agent_db.settings or {}
        system_type = settings.get("system_type", "ncf")
        capabilities = settings.get("capabilities", [])
        if not capabilities:
            capabilities = ["ncf", "rag"] if system_type == "ncf" else ["ceaf", "adaptive_memory"]

        response_list.append(MyAgentResponse(
            agent_id=agent_db.id,
            name=agent_db.name,
            persona=agent_db.persona,
            detailed_persona=agent_db.detailed_persona,
            created_at=agent_db.created_at.isoformat(),
            is_public=agent_db.is_public,  # This is the private/public status of the base agent
            is_public_template=agent_db.is_public_template,
            owner_username=agent_db.user.username if agent_db.user else "Me",
            settings=settings,
            capabilities=capabilities,
            avatar_url=agent_db.avatar_url,
            version=agent_db.version,
            clone_count=agent_db.clone_count,
            usage_cost=agent_db.usage_cost,
            model=agent_db.model  # <-- ADD THIS LINE
        ))
    return response_list


@app.delete("/agents/template/{template_id}/unpublish", tags=["Developer Hub"])
async def unpublish_agent_template(template_id: str, current_user: dict = Depends(verify_token)):
    """Removes a public agent template from the marketplace."""
    with db_repo.SessionLocal() as session:
        agent_template = session.query(Agent).filter(
            Agent.id == template_id,
            Agent.user_id == current_user["user_id"],
            Agent.is_public_template == 1
        ).first()

        if not agent_template:
            raise HTTPException(status_code=404, detail="Public agent template not found or you are not the owner.")

        session.delete(agent_template)
        session.commit()

        # Also remove the agent files from storage
        try:
            public_path = agent_manager.base_storage_path / current_user["user_id"] / template_id
            if public_path.exists():
                shutil.rmtree(public_path)
            # Also remove the config file
            config_file = agent_manager.base_storage_path / current_user["user_id"] / f"{template_id}.json"
            if config_file.exists():
                config_file.unlink()
            # Reload configs in agent_manager
            agent_manager._load_agent_configs()
        except Exception as e:
            logger.error(f"Could not clean up files for unpublished agent {template_id}: {e}")

    return {"message": "Agent has been unpublished and removed from the marketplace."}


@app.put("/agents/template/{template_id}/price", tags=["Developer Hub"])
async def set_agent_price(template_id: str, request: SetPriceRequest, current_user: dict = Depends(verify_token)):
    """Sets the usage cost for a public agent template."""
    with db_repo.SessionLocal() as session:
        agent_template = session.query(Agent).filter(
            Agent.id == template_id,
            Agent.user_id == current_user["user_id"],
            Agent.is_public_template == 1
        ).first()

        if not agent_template:
            raise HTTPException(status_code=404, detail="Public agent template not found or you are not the owner.")

        agent_template.usage_cost = request.usage_cost
        session.commit()

    return {"message": f"Price for agent '{agent_template.name}' updated to {request.usage_cost} credits."}


@app.get("/agents/featured", response_model=List[AgentResponse])
async def get_featured_agents(limit: int = 8):
    """Get featured public agents for discover page"""
    with db_repo.SessionLocal() as session:
        featured_agents = session.query(Agent).filter(
            Agent.is_public_template == 1
        ).order_by(desc(Agent.created_at)).limit(limit).all()

        response_list = []
        for agent in featured_agents:
            settings = agent.settings or {}
            system_type = settings.get("system_type", "ncf")
            capabilities = settings.get("capabilities", [])
            if not capabilities:
                capabilities = ["ncf", "rag"] if system_type == "ncf" else ["ceaf", "adaptive_memory"]

            response_list.append(AgentResponse(
                agent_id=agent.id,
                name=agent.name,
                persona=agent.persona,
                detailed_persona=agent.detailed_persona,
                created_at=agent.created_at.isoformat(),
                is_public=True,  # It is a public template
                owner_username="Public",  # Simpler for featured view
                capabilities=capabilities,
                avatar_url=agent.avatar_url,
                version=agent.version
            ))

        return response_list


@app.post("/admin/create-tensorart-specialist", tags=["Admin"], response_model=dict)
async def create_tensorart_specialist_agent(
        name: str = "TensorArt Specialist",
        current_user: dict = Depends(verify_token)
):
    """
    Creates or verifies the dedicated Tensor.Art specialist agent using its hard-coded ID.
    This is a required, one-time setup step for the system.
    """
    try:
        # ==================== START OF FIX ====================
        # Use the predefined, hard-coded ID instead of generating a new one.
        agent_id = TENSORART_SPECIALIST_AGENT_ID

        # Check if the agent's config file already exists to prevent duplication
        if agent_id in agent_manager._agent_configs:
            return {
                "message": "Tensor.Art specialist agent already exists.",
                "agent_id": agent_id,
                "status": "verified"
            }

        # The agent_manager's create method generates a NEW ID, which is what we need to avoid.
        # We will manually perform the creation steps using the correct ID.

        # 1. Manually create the AgentConfig
        user_id = current_user["user_id"]
        agent_path = AGENT_STORAGE_PATH / user_id / agent_id
        agent_path.mkdir(parents=True, exist_ok=True)
        memory_path = str(agent_path / "memory_blossom.json")

        config = AgentConfig(
            agent_id=agent_id,
            user_id=user_id,
            name=name,
            persona="An AI expert in image generation using the Tensor.Art API.",
            detailed_persona=TENSORART_SPECIALIST_INSTRUCTION,
            model="openrouter/openai/gpt-4o-mini",
            memory_path=memory_path,
            created_at=datetime.now(),
            settings={"system_type": "ncf"}
        )

        # 2. Save the configuration file
        agent_manager._save_agent_config(config)
        agent_manager._agent_configs[agent_id] = config  # Add to in-memory cache

        # 3. Save to the database
        db_repo.create_agent(
            agent_id=agent_id,  # Use the correct, hard-coded ID
            user_id=user_id,
            name=name,
            persona="An AI expert in image generation using the Tensor.Art API.",
            detailed_persona=TENSORART_SPECIALIST_INSTRUCTION,
            model="openrouter/openai/gpt-4o-mini",
            is_public=False
        )

        return {
            "message": "Tensor.Art specialist agent created successfully with the correct system ID.",
            "agent_id": agent_id,
            "status": "created"
        }
        # ===================== END OF FIX =====================
    except Exception as e:
        logger.error(f"Error creating specialist agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/users/{user_id}/add-credits", tags=["Admin"])
async def add_credits_to_user(user_id: str, request: AddCreditsRequest, current_user: dict = Depends(verify_token)):
    # Simple admin check - in production, you'd have a proper role system.
    if current_user["username"] != "xupeta":  # Change "admin" to your admin username
        raise HTTPException(status_code=403, detail="Admin access required")

    with db_repo.SessionLocal() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.credits += request.amount

        transaction = CreditTransaction(
            user_id=user.id,
            amount=request.amount,
            description=request.description
        )
        session.add(transaction)
        session.commit()

        return {
            "message": f"Successfully added {request.amount} credits to user {user.username}. New balance: {user.credits}"}


@app.post("/agents/{agent_id}/avatar", tags=["Agent Management"])
async def upload_agent_avatar(
        agent_id: str,
        file: UploadFile = File(...),
        current_user: dict = Depends(verify_token)
):
    """Uploads or replaces an agent's profile picture."""
    agent_config = agent_manager._agent_configs.get(agent_id)
    if not agent_config or agent_config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Agent not found or unauthorized")

    # Criar o caminho para o avatar
    avatar_dir = AGENT_STORAGE_PATH / agent_config.user_id / agent_id / "avatar"
    avatar_dir.mkdir(parents=True, exist_ok=True)

    # Validar tipo de arquivo (imagem)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Salvar o arquivo (ex: profile.png)
    file_extension = Path(file.filename).suffix or ".png"
    avatar_path = avatar_dir / f"profile{file_extension}"

    with open(avatar_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # URL relativa que o frontend pode usar
    avatar_url = f"/agent_files/{agent_config.user_id}/{agent_id}/avatar/profile{file_extension}"

    # Atualizar o banco de dados
    with db_repo.SessionLocal() as session:
        db_agent = session.query(Agent).filter(Agent.id == agent_id).first()
        if db_agent:
            db_agent.avatar_url = avatar_url
            session.commit()

    return {"message": "Avatar updated successfully", "avatar_url": avatar_url}


@app.post("/agents/{agent_id}/files", tags=["Agent Files & RAG"])
async def upload_file_for_agent(
        agent_id: str,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        current_user: dict = Depends(verify_token)
):
    """Uploads a file to the agent's personal storage for RAG."""
    agent_config = agent_manager._agent_configs.get(agent_id)
    if not agent_config or agent_config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Agent not found or unauthorized")

    files_dir = AGENT_STORAGE_PATH / agent_config.user_id / agent_id / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    file_path = files_dir / file.filename
    if file_path.exists():
        raise HTTPException(status_code=409, detail=f"File '{file.filename}' already exists.")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    agent_storage_path = AGENT_STORAGE_PATH / current_user["user_id"] / agent_id
    background_tasks.add_task(process_and_index_file, agent_storage_path, file_path)

    return {"message": f"File '{file.filename}' uploaded successfully. It will be indexed shortly."}


@app.get("/agents/{agent_id}/files", response_model=List[FileResponseModel], tags=["Agent Files & RAG"])
async def list_agent_files(
        agent_id: str,
        current_user: dict = Depends(verify_token)
):
    """Lists all files available in the agent's personal storage."""
    agent_config = agent_manager._agent_configs.get(agent_id)
    if not agent_config or agent_config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Agent not found or unauthorized")

    files_dir = AGENT_STORAGE_PATH / agent_config.user_id / agent_id / "files"
    if not files_dir.exists():
        return []

    file_list = []
    for f in files_dir.iterdir():
        if f.is_file():  # Ignora a pasta vector_store
            # Tenta determinar o content-type
            import mimetypes
            content_type, _ = mimetypes.guess_type(f.name)

            file_list.append(FileResponseModel(
                filename=f.name,
                content_type=content_type or "application/octet-stream",
                size=f.stat().st_size,
                url=f"/agent_files/{agent_config.user_id}/{agent_id}/files/{f.name}"
            ))

    return file_list


@app.post("/agents/{agent_id}/biography/add", response_model=dict, tags=["Agent Management", "Memory Management"])
async def add_biographical_memories(
        agent_id: str,
        request: AddBiographyRequest,
        current_user: dict = Depends(verify_token)
):
    """
    Adds new "biographical" or "founding" memories to an existing agent.
    This is ideal for enriching an agent's core personality after creation.
    """
    agent_instance = agent_manager.get_agent_instance(agent_id)
    if not agent_instance:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent_instance.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied. You can only modify your own agents.")

    memories_to_add = request.biography
    if not memories_to_add:
        raise HTTPException(status_code=400, detail="The 'biography' list cannot be empty.")

    memories_added = 0
    errors = []
    try:
        for i, memory_data in enumerate(memories_to_add):
            try:
                # Add a special tag to identify these as core biographical memories
                metadata = memory_data.custom_metadata or {}
                metadata.setdefault('source', 'biography_update')
                metadata.setdefault('is_founding_memory', True)

                agent_instance.memory_blossom.add_memory(
                    content=memory_data.content,
                    memory_type=memory_data.memory_type,
                    emotion_score=memory_data.emotion_score,
                    initial_salience=memory_data.initial_salience,
                    custom_metadata=metadata
                )
                memories_added += 1
            except Exception as e:
                errors.append(f"Memory #{i + 1}: {str(e)}")

        if memories_added > 0:
            agent_instance.memory_blossom.save_memories()

        return {
            "agent_id": agent_id,
            "message": "Biographical memories added successfully.",
            "memories_added": memories_added,
            "errors": errors
        }

    except Exception as e:
        logger.error(f"Error adding biographical memories to agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add memories: {str(e)}")


@app.get("/chat/history/agent/{agent_id}", response_model=ChatHistoryResponse, tags=["Chat History"])
async def get_unified_agent_chat_history(
        agent_id: str,
        limit: int = 100,  # Limit the total history to avoid performance issues
        offset: int = 0,
        current_user: dict = Depends(verify_token)
):
    """
    Retrieves the complete, unified chat history between a user and an agent
    by combining all their past sessions.
    """
    try:
        with db_repo.SessionLocal() as session:
            # 1. Verify the user has access to this agent
            agent = session.query(Agent).filter(Agent.id == agent_id).first()
            if not agent or (
                    agent.user_id != current_user['user_id'] and not agent.is_public and not agent.is_public_template):
                raise HTTPException(status_code=403, detail="Agent not found or access denied")

            # 2. Find all session IDs for this user-agent pair
            session_ids_query = session.query(ChatSession.id).filter(
                and_(
                    ChatSession.user_id == current_user['user_id'],
                    ChatSession.agent_id == agent_id
                )
            )
            session_ids = [s_id for s_id, in session_ids_query.all()]

            if not session_ids:
                # No history exists, return an empty response
                return ChatHistoryResponse(
                    session_id="", messages=[], total_messages=0,
                    created_at=datetime.utcnow().isoformat(), last_active=datetime.utcnow().isoformat()
                )

            # 3. Fetch all messages from all those sessions
            messages_query = session.query(Message).filter(
                Message.session_id.in_(session_ids)
            ).order_by(Message.created_at)

            total_messages = messages_query.count()
            # Apply pagination to the final, unified list
            messages = messages_query.offset(offset).limit(limit).all()

            # 4. Format the response
            message_list = [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                    "timestamp": int(msg.created_at.timestamp() * 1000)
                }
                for msg in messages
            ]

            # Find the most recent session to return its metadata
            most_recent_session = session.query(ChatSession).filter(
                ChatSession.id.in_(session_ids)
            ).order_by(desc(ChatSession.last_active)).first()

            return ChatHistoryResponse(
                session_id=most_recent_session.id if most_recent_session else "",
                messages=message_list,
                total_messages=total_messages,
                created_at=most_recent_session.created_at.isoformat() if most_recent_session else "",
                last_active=most_recent_session.last_active.isoformat() if most_recent_session else ""
            )

    except Exception as e:
        logger.error(f"Error getting unified chat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get unified chat history")


# --- Agent Management Routes ---
@app.post("/agents/create", response_model=dict, tags=["Agent Management"])
async def create_agent(request: CreateAgentRequest, current_user: dict = Depends(verify_token)):
    """Creates a new agent (NCF or CEAF) from basic parameters."""
    try:
        agent_id = agent_manager.create_agent(
            user_id=current_user["user_id"],
            name=request.name,
            persona=request.persona,
            detailed_persona=request.detailed_persona,
            model=request.model,
            system_type=request.system_type
        )
        db_repo.create_agent(
            agent_id=agent_id, user_id=current_user["user_id"], name=request.name,
            persona=request.persona, detailed_persona=request.detailed_persona,
            model=request.model, is_public=False
        )
        return {"agent_id": agent_id,
                "message": f"Agent '{request.name}' ({request.system_type}) created successfully."}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error creating agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/create/from-biography", response_model=dict, tags=["Agent Management"])
async def create_agent_from_biography(
        file: UploadFile = File(..., description="A JSON file containing the agent's 'config' and 'biography'."),
        current_user: dict = Depends(verify_token)
):
    """Creates a new agent with a rich, pre-defined biography from a JSON file."""
    if file.content_type != "application/json":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JSON file.")
    try:
        content = await file.read()
        data = json.loads(content)
        validated_data = AgentBiography(**data)
        config_data = validated_data.config
        biography_data = validated_data.biography
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in the uploaded file.")
    except Exception as pydantic_error:
        raise HTTPException(status_code=400, detail=f"Invalid biography file structure: {pydantic_error}")
    try:
        agent_id = agent_manager.create_agent(
            user_id=current_user["user_id"], name=config_data.name, persona=config_data.persona,
            detailed_persona=config_data.detailed_persona, model=config_data.model, system_type=config_data.system_type
        )
        db_repo.create_agent(
            agent_id=agent_id, user_id=current_user["user_id"], name=config_data.name, persona=config_data.persona,
            detailed_persona=config_data.detailed_persona, model=config_data.model, is_public=config_data.is_public
        )
        agent_instance = agent_manager.get_agent_instance(agent_id)
        if not agent_instance:
            raise HTTPException(status_code=500, detail="Failed to instantiate agent after creation.")
        memories_added = 0
        for memory in biography_data:
            agent_instance.memory_blossom.add_memory(**memory.dict())
            memories_added += 1
        if memories_added > 0:
            agent_instance.memory_blossom.save_memories()
        return {
            "agent_id": agent_id, "name": config_data.name, "message": "Agent created successfully from biography.",
            "biography_memories_injected": memories_added
        }
    except Exception as e:
        logger.error(f"Error creating agent from biography: {e}", exc_info=True)
        if 'agent_id' in locals():
            agent_manager.delete_agent(agent_id, current_user["user_id"])
            db_repo.delete_agent(agent_id, current_user["user_id"])
        raise HTTPException(status_code=500, detail=f"Failed to create agent from biography: {str(e)}")


@app.get("/agents/list", response_model=List[AgentResponse], tags=["Agent Management"])
async def list_agents(current_user: dict = Depends(verify_token)):
    """Lists all agents belonging to the current authenticated user."""
    configs = agent_manager.list_user_agents(current_user["user_id"])
    db_agents_map = {agent.id: agent for agent in db_repo.list_user_agents(current_user["user_id"])}
    response_list = []

    for config in configs:
        db_agent = db_agents_map.get(config.agent_id)

        system_type = config.settings.get("system_type", "ncf")
        capabilities = config.settings.get("capabilities", [])

        if not capabilities:
            if system_type == "ceaf":
                capabilities = [
                    "adaptive_memory_architecture", "metacognitive_control_loop",
                    "narrative_coherence_identity", "universal_reflective_analyzer"
                ]
            else:  # Default to NCF
                capabilities = [
                    "narrative_foundation", "rag", "reflector", "isolated_memory"
                ]
        # ===============================================================

        response_list.append(
            AgentResponse(
                agent_id=config.agent_id,
                name=config.name,
                persona=config.persona,

                detailed_persona=config.detailed_persona,
                created_at=config.created_at.isoformat(),
                is_public=bool(db_agent.is_public if db_agent else False),
                owner_username=current_user["username"],
                settings=config.settings,

                capabilities=capabilities,
                avatar_url=db_agent.avatar_url if db_agent else None

            )
        )
    return response_list


@app.get("/agents/public", response_model=List[AgentResponse], tags=["Agent Management"])
async def list_public_agents():
    """Lists the public agent templates available in the marketplace."""
    with db_repo.SessionLocal() as session:
        public_agents_db = session.query(Agent).filter(
            Agent.is_public_template == 1
        ).options(
            joinedload(Agent.user)
        ).order_by(desc(Agent.created_at)).all()

    response_list = []
    for agent_db in public_agents_db:
        # Build the response primarily from the database record for robustness.
        # This ensures all published agents are always included, even if not in the live server cache.

        settings = agent_db.settings or {}
        system_type = settings.get("system_type", "ncf")
        capabilities = settings.get("capabilities", [])

        # Fallback to derive capabilities if they are not explicitly set
        if not capabilities:
            if system_type == "ceaf":
                capabilities = ["ceaf", "adaptive_memory", "evolving_identity"]
            else:  # Default to NCF
                capabilities = ["ncf", "rag", "memory_blossom", "reflector"]

        response_list.append(AgentResponse(
            agent_id=agent_db.id,
            name=agent_db.name,
            persona=agent_db.persona,
            detailed_persona=agent_db.detailed_persona,
            created_at=agent_db.created_at.isoformat(),
            is_public=True,  # This is a list of public templates
            owner_username=agent_db.user.username if agent_db.user else "Unknown",
            settings=settings,
            capabilities=capabilities,
            avatar_url=agent_db.avatar_url,
            version=agent_db.version
        ))

    return response_list


@app.get("/agents/{agent_id}/identity-history", response_model=IdentityHistoryResponse,
         tags=["Agent Management", "CEAF"])
async def get_agent_identity_history(
        agent_id: str,
        current_user: dict = Depends(verify_token)
):
    """
    Retrieves the identity evolution history for a CEAF agent.
    This endpoint is only valid for agents with system_type 'ceaf'.
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verificar permissões (dono ou agente público)
    config = agent.config
    if config.user_id != current_user["user_id"]:
        db_agent = db_repo.get_agent(agent_id)
        if not (db_agent and db_agent.is_public):
            raise HTTPException(status_code=403, detail="Access denied")

    # --- VERIFICAÇÃO CRÍTICA: SÓ FUNCIONA PARA CEAF ---
    if not isinstance(agent, CEAFAgentAdapter):
        raise HTTPException(
            status_code=400,
            detail="This agent is not a CEAF agent and does not have an evolving identity history."
        )

    try:
        # Acessa o histórico através do adapter e do sistema CEAF
        history = agent.ceaf_system.ncim.get_identity_history()

        return IdentityHistoryResponse(
            agent_id=agent_id,
            system_type="ceaf",
            history=history
        )
    except Exception as e:
        logger.error(f"Error getting identity history for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve identity history.")


@app.get("/agents/{agent_id}", response_model=AgentResponse, tags=["Agent Management"])
async def get_agent(agent_id: str, current_user: dict = Depends(verify_token)):
    config = agent_manager._agent_configs.get(agent_id)
    if not config:
        raise HTTPException(status_code=404, detail="Agent not found in file system.")

    db_agent = db_repo.get_agent(agent_id)
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found in database.")

    is_owner = (config.user_id == current_user['user_id'])
    # --- START OF FIX: Allow any authenticated user to view public templates ---
    if not is_owner and not db_agent.is_public_template:
        raise HTTPException(status_code=403, detail="You do not have permission to access this agent.")
    # --- END OF FIX ---

    owner = db_repo.get_user_by_id(config.user_id)

    system_type = config.settings.get("system_type", "ncf")
    capabilities = config.settings.get("capabilities", [])

    if not capabilities:
        if system_type == "ceaf":
            capabilities = [
                "adaptive_memory_architecture", "metacognitive_control_loop",
                "narrative_coherence_identity", "universal_reflective_analyzer"
            ]
        else:
            capabilities = ["narrative_foundation", "rag", "reflector", "isolated_memory"]

    return AgentResponse(
        agent_id=config.agent_id,
        name=config.name,
        persona=config.persona,
        detailed_persona=config.detailed_persona,
        created_at=config.created_at.isoformat(),
        is_public=bool(db_agent.is_public),
        owner_username=owner.username if owner else "Unknown",
        settings=config.settings,
        capabilities=capabilities,
        avatar_url=db_agent.avatar_url if db_agent else None,
        version=db_agent.version if db_agent else "1.0.0",
        clone_count=db_agent.clone_count if db_agent else 0
    )


# EndPoint Live Memory
@app.put("/agents/{agent_id}/live-memory-permissions", tags=["Agent Management"])
async def update_live_memory_permissions(
        agent_id: str,
        request: UpdateLiveMemoryPermissionsRequest,
        current_user: dict = Depends(verify_token)
):
    """Ativa ou desativa as permissões de contribuição e influência da Memória-Live para um agente."""
    agent_config = agent_manager._agent_configs.get(agent_id)
    if not agent_config or agent_config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Agent not found or unauthorized")

    # Atualiza o objeto de configuração em memória
    agent_config.allow_live_memory_contribution = request.allow_contribution
    agent_config.allow_live_memory_influence = request.allow_influence

    # Salva a configuração atualizada no arquivo JSON do agente
    agent_manager._save_agent_config(agent_config)

    return {
        "message": "Live Memory permissions updated successfully.",
        "agent_id": agent_id,
        "contribution_enabled": agent_config.allow_live_memory_contribution,
        "influence_enabled": agent_config.allow_live_memory_influence
    }


app.put("/agents/{agent_id}", response_model=dict, tags=["Agent Management"])


async def update_agent(agent_id: str, request: UpdateAgentRequest, current_user: dict = Depends(verify_token)):
    """Update an existing agent"""
    agent_config = agent_manager._agent_configs.get(agent_id)
    if not agent_config or agent_config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Agent not found or unauthorized")

    # Update file system config
    update_data = request.dict(exclude_unset=True)
    config_updated = False
    for key, value in update_data.items():
        if hasattr(agent_config, key):
            setattr(agent_config, key, value)
            config_updated = True

    if config_updated:
        agent_manager._save_agent_config(agent_config)

    # Update database
    with db_repo.SessionLocal() as session:
        # ======================== FIX #3: CORRECTED QUERY SYNTAX (Bonus fix) ========================
        db_agent = session.query(Agent).filter(Agent.id == agent_id).first()
        # =========================================================================================
        if db_agent:
            if request.name is not None: db_agent.name = request.name
            if request.persona is not None: db_agent.persona = request.persona
            if request.detailed_persona is not None: db_agent.detailed_persona = request.detailed_persona
            if request.avatar_url is not None: db_agent.avatar_url = request.avatar_url
            if request.is_public is not None: db_agent.is_public = 1 if request.is_public else 0
            if request.settings is not None: db_agent.settings = request.settings
            if request.model is not None: db_agent.model = request.model
            session.commit()

    return {"message": "Agent updated successfully"}


# --- ENDPOINTS PARA CHAT HISTORY ---

@app.post("/chat/get-or-create-session", tags=["Chat History"])
async def get_or_create_session(
        agent_id: str,
        current_user: dict = Depends(verify_token)
):
    """
    Gets an active session for the agent or creates a new one if none exists.
    This is now the authoritative source for session management.
    """
    try:
        with db_repo.SessionLocal() as session:
            # --- START FIX: Server-side session restoration logic ---

            # Verify agent exists and user has access
            agent = session.query(Agent).filter(Agent.id == agent_id).first()
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            is_owner = (agent.user_id == current_user['user_id'])
            if not is_owner and not agent.is_public and not agent.is_public_template:
                raise HTTPException(status_code=403, detail="No permission to access this agent")

            # Look for a recent, active session (e.g., last active within 24 hours)
            from datetime import datetime, timedelta
            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            active_session = session.query(ChatSession).filter(
                and_(
                    ChatSession.user_id == current_user['user_id'],
                    ChatSession.agent_id == agent_id,
                    ChatSession.last_active > cutoff_time
                )
            ).order_by(desc(ChatSession.last_active)).first()

            if active_session:
                # If a recent session is found, restore it
                active_session.last_active = datetime.utcnow()
                session.commit()
                return {
                    "session_id": active_session.id,
                    "agent_id": agent_id,
                    "message": "Active session restored from server."
                }
            else:
                # If no recent session exists, create a new one
                new_session = ChatSession(
                    user_id=current_user['user_id'],
                    agent_id=agent_id,
                    created_at=datetime.utcnow(),
                    last_active=datetime.utcnow()
                )
                session.add(new_session)
                session.commit()
                session.refresh(new_session)

                return {
                    "session_id": new_session.id,
                    "agent_id": agent_id,
                    "message": "New session created on server."
                }
            # --- END FIX ---

    except Exception as e:
        logger.error(f"Error in get_or_create_session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get or create session")


@app.post("/chat/save-message", tags=["Chat History"])
async def save_message(
        request: SaveMessageRequest,
        current_user: dict = Depends(verify_token)
):
    """
    Salva uma mensagem no histórico da sessão.
    Deve ser chamado após cada mensagem (user e assistant).
    """
    try:
        with db_repo.SessionLocal() as session:
            # Verificar se a sessão existe e pertence ao usuário
            chat_session = session.query(ChatSession).filter(
                and_(
                    ChatSession.id == request.session_id,
                    ChatSession.user_id == current_user['user_id']
                )
            ).first()

            if not chat_session:
                raise HTTPException(status_code=404, detail="Session not found or no permission")

            # Salvar a mensagem
            new_message = Message(
                session_id=request.session_id,
                role=request.role,
                content=request.content,
                created_at=datetime.utcnow()
            )
            session.add(new_message)

            # Atualizar last_active da sessão
            chat_session.last_active = datetime.utcnow()

            session.commit()
            session.refresh(new_message)

            return {
                "message_id": new_message.id,
                "session_id": request.session_id,
                "status": "saved"
            }

    except Exception as e:
        logger.error(f"Error saving message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save message")


@app.get("/chat/history/{session_id}", response_model=ChatHistoryResponse, tags=["Chat History"])
async def get_chat_history(
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        current_user: dict = Depends(verify_token)
):
    """
    Recupera o histórico de mensagens de uma sessão.
    """
    try:
        with db_repo.SessionLocal() as session:
            # Verificar se a sessão existe e pertence ao usuário
            chat_session = session.query(ChatSession).filter(
                and_(
                    ChatSession.id == session_id,
                    ChatSession.user_id == current_user['user_id']
                )
            ).first()

            if not chat_session:
                raise HTTPException(status_code=404, detail="Session not found or no permission")

            # Buscar mensagens
            messages_query = session.query(Message).filter(
                Message.session_id == session_id
            ).order_by(Message.created_at)

            total_messages = messages_query.count()
            messages = messages_query.offset(offset).limit(limit).all()

            # Converter para formato de resposta
            message_list = [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                    "timestamp": int(msg.created_at.timestamp() * 1000)  # Para compatibilidade com frontend
                }
                for msg in messages
            ]

            return ChatHistoryResponse(
                session_id=session_id,
                messages=message_list,
                total_messages=total_messages,
                created_at=chat_session.created_at.isoformat(),
                last_active=chat_session.last_active.isoformat()
            )

    except Exception as e:
        logger.error(f"Error getting chat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get chat history")


@app.get("/chat/sessions", tags=["Chat History"])
async def get_user_sessions(
        agent_id: Optional[str] = None,
        limit: int = 20,
        current_user: dict = Depends(verify_token)
):
    """
    Lista todas as sessões do usuário, opcionalmente filtradas por agente.
    """
    try:
        with db_repo.SessionLocal() as session:
            # Query base
            query = session.query(ChatSession).filter(
                ChatSession.user_id == current_user['user_id']
            ).options(joinedload(ChatSession.agent))

            # Filtrar por agente se especificado
            if agent_id:
                query = query.filter(ChatSession.agent_id == agent_id)

            # Ordenar por última atividade
            sessions = query.order_by(desc(ChatSession.last_active)).limit(limit).all()

            # Preparar resposta
            session_list = []
            for chat_session in sessions:
                # Contar mensagens
                message_count = session.query(Message).filter(
                    Message.session_id == chat_session.id
                ).count()

                session_list.append(SessionResponse(
                    session_id=chat_session.id,
                    agent_id=chat_session.agent_id,
                    agent_name=chat_session.agent.name if chat_session.agent else "Unknown",
                    created_at=chat_session.created_at.isoformat(),
                    last_active=chat_session.last_active.isoformat(),
                    message_count=message_count
                ))

            return {
                "sessions": session_list,
                "total": len(session_list)
            }

    except Exception as e:
        logger.error(f"Error getting user sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get user sessions")


@app.delete("/chat/session/{session_id}", tags=["Chat History"])
async def delete_session(
        session_id: str,
        current_user: dict = Depends(verify_token)
):
    """
    Deleta uma sessão e todas suas mensagens.
    """
    try:
        with db_repo.SessionLocal() as session:
            # Verificar se a sessão existe e pertence ao usuário
            chat_session = session.query(ChatSession).filter(
                and_(
                    ChatSession.id == session_id,
                    ChatSession.user_id == current_user['user_id']
                )
            ).first()

            if not chat_session:
                raise HTTPException(status_code=404, detail="Session not found or no permission")

            # Deletar sessão (mensagens serão deletadas em cascata)
            session.delete(chat_session)
            session.commit()

            return {
                "message": "Session deleted successfully",
                "session_id": session_id
            }

    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete session")


@app.delete("/agents/{agent_id}", tags=["Agent Management"])
async def delete_agent(agent_id: str, current_user: dict = Depends(verify_token)):
    success = agent_manager.delete_agent(agent_id, current_user["user_id"])
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found or unauthorized")
    db_repo.delete_agent(agent_id, current_user["user_id"])
    return {"message": "Agent deleted successfully"}


# --- Enhanced Agent Profile Management ---
@app.put("/agents/{agent_id}/profile", response_model=dict)
async def update_agent_profile(
        agent_id: str,
        request: EnhancedUpdateAgentRequest,
        current_user: dict = Depends(verify_token)
):
    """
    Enhanced agent profile update with comprehensive editing capabilities
    Allows editing: name, persona (short description), detailed_persona, model, and settings
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # Track what was updated
        updated_fields = []

        # Update agent configuration
        config = agent.config

        if request.model is not None:
            old_model = config.model
            config.model = request.model
            agent.model = LiteLlm(model=config.model)
            if hasattr(agent, 'adk_agent'):
                agent.adk_agent.model = agent.model
            updated_fields.append(f"model: '{old_model}' -> '{request.model}'")

        if request.name is not None:
            old_name = config.name
            config.name = request.name
            updated_fields.append(f"name: '{old_name}' → '{request.name}'")

        if request.persona is not None:
            old_persona = config.persona
            config.persona = request.persona
            updated_fields.append(f"persona: '{old_persona[:50]}...' → '{request.persona[:50]}...'")

        if request.detailed_persona is not None:
            old_detailed = config.detailed_persona
            config.detailed_persona = request.detailed_persona
            updated_fields.append(f"detailed_persona: '{old_detailed[:50]}...' → '{request.detailed_persona[:50]}...'")

        if request.settings is not None:
            config.settings.update(request.settings)
            updated_fields.append(f"settings: {list(request.settings.keys())}")

        # Save updated configuration to file system
        agent_manager._save_agent_config(config)

        # Update in database if it exists
        try:
            with db_repo.SessionLocal() as session:
                db_agent = session.query(Agent).filter(Agent.id == agent_id).first()
                if db_agent:
                    if request.name is not None:
                        db_agent.name = request.name
                    if request.persona is not None:
                        db_agent.persona = request.persona
                    if request.detailed_persona is not None:
                        db_agent.detailed_persona = request.detailed_persona
                    if request.avatar_url is not None:
                        db_agent.avatar_url = request.avatar_url
                    if request.is_public is not None:
                        db_agent.is_public = 1 if request.is_public else 0
                    if request.settings is not None:
                        db_agent.settings = request.settings

                    session.commit()
                    updated_fields.append("database record")
        except Exception as db_error:
            logger.warning(f"Database update failed: {db_error}")
            # Continue execution - file system update is more critical

        # Add memory about the profile update
        try:
            update_memory = {
                "content": f"Agent profile updated. Changed: {', '.join(updated_fields)}",
                "memory_type": "system_update",
                "emotion_score": 0.0,
                "initial_salience": 0.3,
                "custom_metadata": {
                    "source": "profile_update",
                    "updated_by": current_user["user_id"],
                    "timestamp": datetime.now().isoformat(),
                    "updated_fields": updated_fields
                }
            }

            agent.memory_blossom.add_memory(**update_memory)
            agent.memory_blossom.save_memories()
        except Exception as memory_error:
            logger.warning(f"Failed to add update memory: {memory_error}")

        return {
            "message": "Agent profile updated successfully",
            "agent_id": agent_id,
            "agent_name": config.name,
            "updated_fields": updated_fields,
            "timestamp": datetime.now().isoformat(),
            "ncf_enabled": True,
            "capabilities": ["ncf", "memory", "narrative_foundation", "rag", "reflector"]
        }

    except Exception as e:
        logger.error(f"Error updating agent profile {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Profile update failed: {str(e)}")


@app.get("/agents/{agent_id}/profile", response_model=dict)
async def get_agent_profile(
        agent_id: str,
        current_user: dict = Depends(verify_token)
):
    """
    Get comprehensive agent profile information for NCF or CEAF agents.
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Access permission check... (unchanged)
    config = agent.config
    if config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied. You can only chat with your own agents.")

    # ======================== THE FIX IS HERE ========================
    # Instead of trying to access .memory_stores, call the get_memory_stats method.
    # The CEAF adapter now has this method and will return the data in the expected format.
    memory_stats = {}
    if hasattr(agent.memory_blossom, 'get_memory_stats'):
        memory_stats = agent.memory_blossom.get_memory_stats()
    else:
        # Fallback for older/simpler memory objects, though our adapter has the method.
        memory_stats = {"total_memories": 0, "memory_types": [], "memory_breakdown": {}}
    # ===============================================================

    return {
        "agent_id": config.agent_id,
        "name": config.name,
        "persona": config.persona,
        "detailed_persona": config.detailed_persona,
        "model": config.model,
        "created_at": config.created_at.isoformat() if config.created_at else None,
        "settings": config.settings or {},
        "memory_stats": memory_stats,  # Use the stats we just fetched
        "allow_live_memory_contribution": config.allow_live_memory_contribution,
        "allow_live_memory_influence": config.allow_live_memory_influence,
        "owner_id": config.user_id,
        "is_owner": config.user_id == current_user["user_id"]
    }


@app.get("/agents/{agent_id}/adaptive-stats")
async def get_agent_adaptive_stats(agent_id: str, current_user: dict = Depends(verify_token)):
    """Get adaptive RAG statistics for an agent"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        adaptive_stats = agent.memory_blossom.get_adaptive_stats()
        return {
            "agent_id": agent_id,
            "agent_name": agent.config.name,
            "adaptive_stats": adaptive_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting adaptive stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get adaptive stats")


# --- NOVO: Endpoint para publicar um agente ---
@app.post("/agents/{agent_id}/publish", tags=["Developer Hub"], status_code=status.HTTP_201_CREATED)
async def publish_agent_to_marketplace(
        agent_id: str,
        changelog: str = Form("First public release."),
        include_chat_history: bool = Form(True),
        current_user: dict = Depends(verify_token)
):
    """
    Publishes a private agent, creating a public template via a smart snapshot.
    Combines original memories, additional memories, and optionally, memories derived from chat history.
    FIXED: Correctly places the new public agent's config file at the user directory level.
    """
    source_config = agent_manager.agent_configs.get(agent_id)
    if not source_config or source_config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Agent not found or unauthorized.")

    public_agent_id = str(uuid.uuid4())
    user_dir = agent_manager.base_storage_path / current_user["user_id"]

    # === FIX START: Correctly define paths ===
    # The config file for the new public agent MUST be directly under the user's directory.
    public_config_file_path = user_dir / f"{public_agent_id}.json"

    # The data directory (for memory, avatars, etc.) is a subdirectory named after the new ID.
    public_data_path = user_dir / public_agent_id
    # === FIX END ===

    source_config_file_path = user_dir / f"{agent_id}.json"
    source_data_dir_path = user_dir / agent_id

    try:
        public_data_path.mkdir(parents=True, exist_ok=True)
        if not source_config_file_path.exists():
            raise FileNotFoundError(f"Critical error: Source config file not found at {source_config_file_path}")

        # Copy the original config to the new, correct location
        shutil.copy2(source_config_file_path, public_config_file_path)

        # Update the copied config file with the new public ID and correct memory path
        with open(public_config_file_path, 'r+') as f:
            config_dict = json.load(f)
            config_dict['agent_id'] = public_agent_id
            # The memory path points to the new data directory
            config_dict['memory_path'] = str(public_data_path / 'memory_blossom.json')
            f.seek(0)
            json.dump(config_dict, f, indent=2)
            f.truncate()

        # Copy the source agent's data directory (memory, avatar etc.) to the new public data directory
        if source_data_dir_path.exists():
            shutil.copytree(source_data_dir_path, public_data_path, dirs_exist_ok=True)

        # --- Memory combination logic (this part was already correct) ---
        combined_memories = []
        source_instance = agent_manager.get_agent_instance(agent_id)
        if source_instance and hasattr(source_instance, 'memory_blossom') and source_instance.memory_blossom:
            all_private_memories = source_instance.memory_blossom.get_all_memories()
            if all_private_memories:
                if hasattr(all_private_memories[0], 'to_dict'):
                    combined_memories.extend([mem.to_dict() for mem in all_private_memories])
                else:
                    combined_memories.extend(all_private_memories)

        if include_chat_history:
            chat_memories = await _convert_chat_history_to_memories(agent_id, current_user["user_id"])
            combined_memories.extend(chat_memories)

        if combined_memories:
            memory_data_to_save = {"memory_stores": {}}
            for mem in combined_memories:
                mem_type = mem.get("memory_type", "Explicit")
                if mem_type not in memory_data_to_save["memory_stores"]:
                    memory_data_to_save["memory_stores"][mem_type] = []
                if 'id' not in mem or mem['id'] is None:
                    mem['id'] = str(uuid.uuid4())
                memory_data_to_save["memory_stores"][mem_type].append(mem)

            public_memory_path = public_data_path / 'memory_blossom.json'
            with open(public_memory_path, 'w') as f:
                json.dump(memory_data_to_save, f, indent=2)

        # --- Database entry (this part was also correct) ---
        with db_repo.SessionLocal() as session:
            public_template = Agent(
                id=public_agent_id, user_id=current_user["user_id"], name=source_config.name,
                persona=source_config.persona, detailed_persona=source_config.detailed_persona,
                model=source_config.model, settings=source_config.settings,
                is_public_template=1, parent_agent_id=agent_id,
                version="1.0.0", changelog=changelog, clone_count=0
            )
            session.add(public_template)
            session.commit()

        # Reload agent manager's cache to include the new public agent
        agent_manager._load_agent_configs()

        return {
            "message": f"Agent '{source_config.name}' published successfully! It's now a public template.",
            "public_template_id": public_agent_id,
            "version": "1.0.0",
            "memories_included": len(combined_memories)
        }
    except Exception as e:
        # Cleanup failed attempt
        if public_config_file_path.exists():
            public_config_file_path.unlink()
        if public_data_path.exists():
            shutil.rmtree(public_data_path)
        logger.error(f"Error publishing agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to publish agent: {str(e)}")


# --- NOVO: Endpoint para atualizar um template público ---
@app.post("/agents/template/{template_id}/update", tags=["Developer Hub"])
async def update_public_agent_template(
        template_id: str,
        changelog: str = Form(...),
        current_user: dict = Depends(verify_token)
):
    """Cria uma NOVA VERSÃO de um template público a partir do estado atual do agente privado do criador."""
    # Lógica similar ao publish, mas incrementando a versão e usando o mesmo parent_agent_id.
    # Esta implementação fica como próximo passo, mas a estrutura seria quase idêntica à de `publish`.
    pass


# --- NOVO: Endpoint para o Hub do Desenvolvedor ---
@app.get("/developer/my-publications", tags=["Developer Hub"])
async def get_my_published_agents(current_user: dict = Depends(verify_token)):
    """Lista todos os templates que o usuário atual publicou."""
    with db_repo.SessionLocal() as session:
        publications = session.query(Agent).filter(
            Agent.user_id == current_user["user_id"],
            Agent.is_public_template == 1
        ).order_by(Agent.name, Agent.version.desc()).all()

        return [
            {
                "public_id": agent.id,
                "name": agent.name,
                "version": agent.version,
                "changelog": agent.changelog,
                "clone_count": agent.clone_count,
                "published_at": agent.created_at.isoformat()
            } for agent in publications
        ]


@app.post("/agents/{agent_id}/upgrade-adaptive-rag")
async def upgrade_agent_adaptive_rag(agent_id: str, current_user: dict = Depends(verify_token)):
    """Upgrade an existing agent to use Enhanced MemoryBlossom"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        from enhanced_memory_system import upgrade_agent_to_adaptive_rag
        success = upgrade_agent_to_adaptive_rag(agent, enable_adaptive_rag=True)

        if success:
            return {
                "message": f"Agent '{agent.config.name}' successfully upgraded to Enhanced MemoryBlossom",
                "adaptive_rag_enabled": True,
                "features_added": [
                    "Domain-aware clustering",
                    "Performance-weighted retrieval",
                    "Adaptive concept formation",
                    "Multi-layer memory system"
                ]
            }
        else:
            raise HTTPException(status_code=500, detail="Upgrade failed")

    except Exception as e:
        logger.error(f"Error upgrading agent: {e}")
        raise HTTPException(status_code=500, detail=f"Upgrade failed: {str(e)}")


# --- Pre-built Agent Routes ---
@app.get("/prebuilt-agents", response_model=List[PrebuiltAgentResponse])
async def list_prebuilt_agents(
        system_type: Optional[str] = None,
        archetype: Optional[str] = None,
        maturity_level: Optional[str] = None,
        min_rating: Optional[float] = None
):
    """List available pre-built agents"""
    try:
        # Convert strings to enums if provided
        archetype_enum = AgentArchetype(archetype) if archetype else None
        maturity_enum = AgentMaturityLevel(maturity_level) if maturity_level else None

        agents = prebuilt_repo.get_available_agents(
            system_type=system_type,
            archetype=archetype_enum,
            maturity_level=maturity_enum
        )

        # Filter by rating if specified
        if min_rating:
            agents = [a for a in agents if a.rating >= min_rating]

        # Convert to response
        response_agents = []
        for agent in agents:
            # Load sample_memories from the file
            sample_memories = []
            try:
                agent_path = prebuilt_repo.storage_path / f"{agent.id}.json"
                if agent_path.exists():
                    with open(agent_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Get first 3 memories as a sample
                        memories = data.get("memories", [])
                        sample_memories = [
                            {
                                "type": mem.get("memory_type", "Unknown"),
                                "content": mem.get("content", "")[:150] + "..." if len(
                                    mem.get("content", "")) > 150 else mem.get("content", ""),
                                "emotion_score": mem.get("emotion_score", 0.0),
                                "salience": mem.get("initial_salience", 0.5)
                            }
                            for mem in memories[:3]
                        ]
            except Exception as e:
                logger.warning(f"Could not load sample memories for agent {agent.id}: {e}")

            response_agents.append(PrebuiltAgentResponse(
                id=agent.id,
                name=agent.name,
                archetype=agent.archetype.value,
                maturity_level=agent.maturity_level.value,
                system_type=agent.system_type,
                short_description=agent.short_description,
                detailed_persona=agent.detailed_persona,
                total_interactions=agent.total_interactions,
                rating=agent.rating,
                download_count=agent.download_count,
                breakthrough_count=agent.breakthrough_count if agent.system_type == "ceaf" else None,
                coherence_average=agent.coherence_average if agent.system_type == "ceaf" else None,
                tags=agent.tags,
                created_by=agent.created_by,
                sample_memories=sample_memories
            ))

        return response_agents

    except Exception as e:
        logger.error(f"Error listing prebuilt agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load prebuilt agents")


@app.get("/prebuilt-agents/list", response_model=List[PrebuiltAgentResponse])
async def list_prebuilt_agents_alt():
    """Alias para compatibilidade com frontend"""
    return await list_prebuilt_agents()


@app.get("/prebuilt-agents/{agent_id}", response_model=PrebuiltAgentResponse)
async def get_prebuilt_agent_details(agent_id: str):
    """Get details of a specific pre-built agent"""
    try:
        if agent_id not in prebuilt_repo.agents:
            raise HTTPException(status_code=404, detail="Prebuilt agent not found")

        agent = prebuilt_repo.agents[agent_id]

        biographical_memories = []
        try:
            agent_path = prebuilt_repo.storage_path / f"{agent.id}.json"
            with open(agent_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                memories = data.get("memories", [])

                for mem in memories:
                    biographical_memories.append({
                        "type": mem.get("memory_type", "Unknown"),
                        "content": mem.get("content", ""),

                    })

        except Exception as e:
            logger.warning(f"Could not load memories for agent {agent.id}: {e}")

        return PrebuiltAgentResponse(
            id=agent.id,
            name=agent.name,
            archetype=agent.archetype.value,
            maturity_level=agent.maturity_level.value,
            system_type=agent.system_type,
            short_description=agent.short_description,
            detailed_persona=agent.detailed_persona,
            total_interactions=agent.total_interactions,
            rating=agent.rating,
            download_count=agent.download_count,
            breakthrough_count=agent.breakthrough_count if agent.system_type == "ceaf" else None,
            coherence_average=agent.coherence_average if agent.system_type == "ceaf" else None,
            tags=agent.tags,
            created_by=agent.created_by,
            sample_memories=biographical_memories
        )


    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prebuilt agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load agent details")


# --- Agent Management Routes ---
@app.post("/agents/ncfcreate", response_model=dict)
async def create_ncf_agent(request: CreateAgentRequest, current_user: dict = Depends(verify_token)):
    try:
        agent_id = agent_manager.create_agent(
            user_id=current_user["user_id"],
            name=request.name,
            persona=request.persona,
            detailed_persona=request.detailed_persona,
            model=request.model,
            system_type="ncf"
        )
        db_repo.create_agent(
            agent_id=agent_id, user_id=current_user["user_id"],
            name=request.name, persona=request.persona,
            detailed_persona=request.detailed_persona, model=request.model, is_public=request.is_public
        )
        return {"agent_id": agent_id, "message": f"NCF agent '{request.name}' created successfully."}
    except Exception as e:
        logger.error(f"Error creating NCF agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== FIX: NEW ENDPOINT ADDED ====================
@app.post("/agents/ceaf/create", response_model=dict)
async def create_ceaf_agent(request: CreateAgentRequest, current_user: dict = Depends(verify_token)):
    """Creates a new CEAF-enabled agent."""
    if not agent_manager.is_ceaf_available():
        raise HTTPException(status_code=501, detail="CEAF system is not available on this server.")

    try:
        # The agent manager handles the specifics of creating a CEAF agent
        agent_id = agent_manager.create_agent(
            user_id=current_user["user_id"],
            name=request.name,
            persona=request.persona,
            detailed_persona=request.detailed_persona,
            model=request.model,
            system_type="ceaf"
        )
        db_repo.create_agent(
            agent_id=agent_id, user_id=current_user["user_id"],
            name=request.name, persona=request.persona,
            detailed_persona=request.detailed_persona, model=request.model, is_public=request.is_public
        )
        return {"agent_id": agent_id, "message": f"CEAF agent '{request.name}' created successfully."}
    except Exception as e:
        logger.error(f"Error creating CEAF agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/clone", response_model=dict, tags=["Agent Management"])
async def clone_agent(request: CloneAgentRequest, current_user: dict = Depends(verify_token)):
    """Clones a pre-built OR a public user-created agent for the current user."""
    try:
        source_agent_id = request.source_agent_id
        clone_data = None

        # Check pre-built repo first
        if source_agent_id in prebuilt_repo.agents:
            clone_data = prebuilt_repo.clone_agent_for_user(
                agent_id=source_agent_id,
                user_id=current_user["user_id"],
                custom_name=request.custom_name
            )
        else:
            # Check if it's a public user-created agent template
            public_agent_config = agent_manager._agent_configs.get(source_agent_id)
            if public_agent_config:
                db_agent = db_repo.get_agent(source_agent_id)

                # FIXED: Check the correct flag 'is_public_template' instead of 'is_public'.
                if db_agent and db_agent.is_public_template:
                    source_instance = agent_manager.get_agent_instance(source_agent_id)
                    initial_memories = []

                    # This part is complex, let's ensure it's robust
                    if source_instance and hasattr(source_instance,
                                                   'memory_blossom') and source_instance.memory_blossom:
                        if hasattr(source_instance.memory_blossom, 'get_all_memories'):
                            initial_memories = source_instance.memory_blossom.get_all_memories()
                        else:  # Fallback
                            initial_memories = source_instance.memory_blossom.retrieve_memories(query="*", top_k=9999)

                    clone_data = {
                        "agent_config": {
                            "name": request.custom_name or f"Copy of {public_agent_config.name}",
                            "persona": public_agent_config.persona,
                            "detailed_persona": public_agent_config.detailed_persona,
                            "system_type": public_agent_config.settings.get("system_type", "ncf"),
                            "model": public_agent_config.model,
                            "archetype": public_agent_config.settings.get("archetype")
                        },
                        "initial_memories": initial_memories,
                        "source_agent_id": source_agent_id
                    }

        if not clone_data:
            raise HTTPException(status_code=404, detail="Source agent not found or is not a public template.")

        agent_config = clone_data["agent_config"]
        initial_memories_raw = clone_data["initial_memories"] if request.clone_memories else []

        sanitized_name = sanitize_agent_name(agent_config["name"])
        logger.info(f"Sanitizing agent name: '{agent_config['name']}' -> '{sanitized_name}'")

        # Create the new agent (your existing logic is fine from here)
        agent_id = agent_manager.create_agent(
            user_id=current_user["user_id"],
            name=sanitized_name,
            persona=agent_config["persona"],
            detailed_persona=agent_config["detailed_persona"],
            model=agent_config.get("model"),
            system_type=agent_config.get("system_type", "ncf")
        )

        # Normalize and inject memories
        normalized_memories = []
        if request.clone_memories and initial_memories_raw:
            if initial_memories_raw and hasattr(initial_memories_raw[0], 'to_dict'):
                normalized_memories = [mem.to_dict() for mem in initial_memories_raw]
            else:
                normalized_memories = initial_memories_raw

        if normalized_memories:
            agent_instance = agent_manager.get_agent_instance(agent_id)
            if agent_instance and hasattr(agent_instance, 'memory_blossom'):
                for memory_dict in normalized_memories:
                    agent_instance.memory_blossom.add_memory(
                        content=memory_dict.get("content", ""),
                        memory_type=memory_dict.get("memory_type", "Explicit"),
                        custom_metadata=memory_dict.get("custom_metadata", {}),
                        emotion_score=memory_dict.get("emotion_score", 0.0),
                        initial_salience=memory_dict.get("salience", 0.5)
                    )
                agent_instance.memory_blossom.save_memories()

        # Create DB record
        with db_repo.SessionLocal() as session:
            db_repo.create_agent(
                agent_id=agent_id, user_id=current_user["user_id"],
                name=sanitized_name, persona=agent_config["persona"],
                detailed_persona=agent_config["detailed_persona"], model=agent_config.get("model"),
                is_public=False  # Cloned agents are private to the new user
            )
            # Increment the clone count of the source template
            source_template = session.query(Agent).filter(Agent.id == source_agent_id).first()
            if source_template:
                source_template.clone_count += 1
                session.commit()

        # Add metadata to config file
        new_agent_instance = agent_manager.get_agent_instance(agent_id)
        if new_agent_instance:
            new_agent_instance.config.settings["cloned_from"] = source_agent_id
            agent_manager._save_agent_config(new_agent_instance.config)

        return {
            "agent_id": agent_id,
            "message": f"Successfully cloned '{agent_config['name']}'. New agent name is '{sanitized_name}'.",
            "system_type": agent_config["system_type"],
            "memories_cloned": len(normalized_memories)
        }
    except Exception as e:
        logger.error(f"Error cloning agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clone agent: {str(e)}")


@app.get("/prebuilt-agents/archetypes")
async def get_available_archetypes():
    """List available archetypes"""
    return {
        "archetypes": [
            {
                "value": archetype.value,
                "name": archetype.value.replace("_", " ").title(),
                "description": get_archetype_description(archetype)
            }
            for archetype in AgentArchetype
        ]
    }


@app.get("/prebuilt-agents/maturity-levels")
async def get_maturity_levels():
    """List available maturity levels"""
    descriptions = {
        AgentMaturityLevel.NEWBORN: "Fresh personality, 0-10 interactions",
        AgentMaturityLevel.LEARNING: "Developing traits, 11-50 interactions",
        AgentMaturityLevel.DEVELOPING: "Growing wisdom, 51-200 interactions",
        AgentMaturityLevel.MATURE: "Established personality, 201-1000 interactions",
        AgentMaturityLevel.EXPERIENCED: "Deep understanding, 1000+ interactions",
        AgentMaturityLevel.MASTER: "Highly refined, exceptional agents"
    }

    return {
        "maturity_levels": [
            {
                "value": level.value,
                "name": level.value.replace("_", " ").title(),
                "description": descriptions[level]
            }
            for level in AgentMaturityLevel
        ]
    }


@app.post("/chat/{agent_id}", tags=["Chat"])
async def chat_with_agent_enhanced(
        agent_id: str,
        background_tasks: BackgroundTasks,  # ADD this
        message: str = Form(...),
        session_id: Optional[str] = Form(None),
        current_user: dict = Depends(verify_token)
):
    try:
        result = await process_chat_message(
            agent_id=agent_id,
            message=message,
            session_id=session_id,
            current_user=current_user,
            background_tasks=background_tasks,  # PASS it down
            save_to_history=True
        )
        return result
    except Exception as e:
        logger.error(f"Error in FormData chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat processing failed")


# --- ENDPOINT RÁPIDO (SEM PERSISTÊNCIA) ---
@app.post("/chat/{agent_id}/quick", tags=["Chat"])
async def quick_chat(
        agent_id: str,
        request: ChatRequest,
        background_tasks: BackgroundTasks,  # ADD this
        current_user: dict = Depends(verify_token)
):
    try:
        result = await process_chat_message(
            agent_id=agent_id,
            message=request.message,
            session_id=request.session_id,
            current_user=current_user,
            background_tasks=background_tasks,  # PASS it down
            save_to_history=False
        )
        return {
            "response": result["response"],
            "session_id": result["session_id"],
            "note": "This conversation was not saved to history"
        }
    except Exception as e:
        logger.error(f"Error in quick chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Quick chat failed")


# --- Chat Routes ---
@app.post("/agents/{agent_id}/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_with_agent(
        agent_id: str,
        request: ChatRequest, # <-- This now includes the overrides
        background_tasks: BackgroundTasks,
        save_history: bool = True,
        current_user: dict = Depends(verify_token)
):
    try:
        result = await process_chat_message(
            agent_id=agent_id,
            message=request.message,
            session_id=request.session_id,
            current_user=current_user,
            background_tasks=background_tasks,
            save_to_history=save_history,
            session_overrides=request.session_overrides # <-- PASS THE OVERRIDES HERE
        )
        return ChatResponse(
            response=result["response"],
            session_id=result["session_id"],
            ncf_enabled=result.get("ncf_enabled", True)
        )
    except Exception as e:
        logger.error(f"Error in JSON chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}"
        )


# --- Memory Management Routes ---
@app.get("/agents/{agent_id}/memories")
async def get_agent_memories(
        agent_id: str,
        memory_type: Optional[str] = None,
        limit: int = 50,
        current_user: dict = Depends(verify_token)
):
    """
    Get memories for a specific agent (works for both NCF and CEAF).
    FIXED: Uses a direct listing method instead of semantic search to show ALL memories.
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # --- INÍCIO DA CORREÇÃO ---
        # Use the new direct method to get all memories instead of searching for "*"
        if hasattr(agent.memory_blossom, 'get_all_memories'):
            all_raw_memories = agent.memory_blossom.get_all_memories()
        else:
            # Fallback for older memory systems that might not have the new method
            all_raw_memories = agent.memory_blossom.retrieve_memories(query="*", top_k=9999)

        # Convert all retrieved items to dictionaries to ensure they are JSON serializable.
        memories_as_dicts = []
        for mem in all_raw_memories:
            if hasattr(mem, 'to_dict') and callable(mem.to_dict):
                memories_as_dicts.append(mem.to_dict())  # Convert NCF Memory object
            elif isinstance(mem, dict):
                memories_as_dicts.append(mem)  # Already a dict (from CEAF adapter)

        # Now, perform any filtering and limiting on the complete list of dictionaries
        final_memories = memories_as_dicts
        if memory_type:
            final_memories = [m for m in final_memories if m.get('memory_type') == memory_type]

        final_memories = final_memories[:limit]
        # --- FIM DA CORREÇÃO ---

        # Get overall stats using the compatible method
        stats = agent.memory_blossom.get_memory_stats()

    except Exception as e:
        logger.error(f"Error retrieving memories for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve memories.")

    return {
        "agent_id": agent_id,
        "agent_name": agent.config.name,
        "total_memories": stats.get("total_memories", 0),
        "memory_types": stats.get("memory_types", []),
        "memories": final_memories
    }


# --- NOVA FUNÇÃO HELPER PARA CONVERTER CHAT EM MEMÓRIAS ---
async def _convert_chat_history_to_memories(agent_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Usa um LLM para analisar o histórico de chat e extrair memórias biográficas."""
    logger.info(f"Converting chat history to memories for agent {agent_id}")
    try:
        with db_repo.SessionLocal() as session:
            # Pega a sessão de chat mais recente para este agente e usuário
            chat_session = session.query(ChatSession).filter(
                ChatSession.agent_id == agent_id,
                ChatSession.user_id == user_id
            ).order_by(desc(ChatSession.last_active)).first()

            if not chat_session:
                logger.warning("No chat session found to derive memories from.")
                return []

            # Pega as últimas 100 mensagens
            messages = session.query(Message).filter(
                Message.session_id == chat_session.id
            ).order_by(Message.created_at.desc()).limit(100).all()

            if len(messages) < 10:  # Não processa conversas muito curtas
                logger.info("Chat history is too short to derive meaningful memories.")
                return []

            # Formata o histórico para o LLM
            transcript = "\n".join([f"{msg.role}: {msg.content}" for msg in reversed(messages)])

            # Prompt para o LLM biógrafo
            biographer_prompt = f"""
            You are an AI Biographer. Your task is to analyze the following conversation transcript and extract 3-5 key, defining "biographical memories" that capture the agent's personality, core beliefs, or significant learned behaviors as demonstrated in these chats.

            RULES:
            - The memories MUST be from the agent's first-person perspective ("I learned that...", "I believe...", "My approach is...").
            - DO NOT include any personal information about the user. Focus only on the agent's emergent persona.
            - Extract general traits and philosophies, not specific facts from one conversation.
            - The output MUST be a valid JSON list of memory objects. Do not include any other text.
            - Choose memory_type from ['Explicit', 'Emotional', 'Procedural'].

            TRANSCRIPT:
            ---
            {transcript}
            ---

            JSON OUTPUT:
            """

            response = await Litellmtop.completion(
                model="openrouter/openai/gpt-4o-mini",  # Um modelo bom e barato para esta tarefa
                messages=[{"role": "user", "content": biographer_prompt}],
                response_format={"type": "json_object"}
            )

            extracted_json = response.choices[0].message.content
            memory_list = json.loads(extracted_json)

            # Validação simples
            if not isinstance(memory_list, list):
                return []

            # Adiciona metadados
            for mem in memory_list:
                mem['custom_metadata'] = {'source': 'chat_history_biographer'}
                mem.setdefault('emotion_score', 0.0)
                mem.setdefault('initial_salience', 0.7)  # Memórias derivadas de chat são importantes

            logger.info(f"Successfully derived {len(memory_list)} memories from chat history.")
            return memory_list

    except Exception as e:
        logger.error(f"Failed to convert chat history to memories: {e}", exc_info=True)
        return []


@app.post("/agents/{agent_id}/memories/search")
async def search_agent_memories(
        agent_id: str,
        request: MemorySearchRequest,
        current_user: dict = Depends(verify_token)
):
    """Search memories for a specific NCF-enabled agent"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    # Search memories using the agent's isolated memory system
    results = agent.memory_blossom.retrieve_memories(
        query=request.query,
        target_memory_types=request.memory_types,
        top_k=request.limit or 10
    )

    return {
        "agent_id": agent_id,
        "agent_name": agent.config.name,
        "query": request.query,
        "isolated_memory_system": True,
        "ncf_enabled": True,
        "results": [mem.to_dict() for mem in results]
    }


@app.delete("/agents/{agent_id}/memories/{memory_id}")
async def delete_memory(
        agent_id: str,
        memory_id: str,
        current_user: dict = Depends(verify_token)
):
    """Delete a specific memory from an NCF-enabled agent"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    # Find and remove the memory from the agent's isolated memory system
    memory_found = False
    for mem_type, mem_list in agent.memory_blossom.memory_stores.items():
        for i, mem in enumerate(mem_list):
            if mem.id == memory_id:
                mem_list.pop(i)
                memory_found = True
                agent.memory_blossom.save_memories()
                break
        if memory_found:
            break

    if not memory_found:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {"message": "Memory deleted successfully from NCF-enabled agent"}


# --- Routes for Model Updates in Chat ---

# --- Enhanced Memory Management Routes ---
@app.get("/agents/{agent_id}/memories/export", response_model=MemoryExportResponse)
async def export_agent_memories(
        agent_id: str,
        format: Optional[str] = "json",  # json, csv, or zip
        memory_types: Optional[str] = None,  # comma-separated list
        current_user: dict = Depends(verify_token)
):
    """
    Export all memories for a specific agent
    Supports JSON, CSV, and ZIP formats
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # Get all memories from the agent's isolated memory system
        all_memories = []
        memory_types_filter = memory_types.split(",") if memory_types else None

        for mem_type, mem_list in agent.memory_blossom.memory_stores.items():
            if memory_types_filter and mem_type not in memory_types_filter:
                continue

            for memory in mem_list:
                memory_dict = memory.to_dict()
                memory_dict['memory_type'] = mem_type  # Ensure type is included
                all_memories.append(memory_dict)

        export_data = {
            "agent_id": agent_id,
            "agent_name": agent.config.name,
            "export_timestamp": datetime.now().isoformat(),
            "total_memories": len(all_memories),
            "memory_types": list(agent.memory_blossom.memory_stores.keys()),
            "memories": all_memories
        }

        if format.lower() == "json":
            return MemoryExportResponse(**export_data)

        elif format.lower() == "csv":
            # Convert to CSV format
            try:
                import pandas as pd
            except ImportError:
                raise HTTPException(status_code=500, detail="pandas not installed. Please install: pip install pandas")

            if not all_memories:
                raise HTTPException(status_code=404, detail="No memories found to export")

            # Flatten memories for CSV
            flattened_memories = []
            for memory in all_memories:
                flat_memory = {
                    'id': memory.get('id'),
                    'content': memory.get('content'),
                    'memory_type': memory.get('memory_type'),
                    'emotion_score': memory.get('emotion_score', 0.0),
                    'coherence_score': memory.get('coherence_score', 0.5),
                    'novelty_score': memory.get('novelty_score', 0.5),
                    'salience': memory.get('salience', 0.5),
                    'created_at': memory.get('created_at'),
                    'accessed_count': memory.get('accessed_count', 0),
                    'last_accessed': memory.get('last_accessed'),
                    'custom_metadata': json.dumps(memory.get('custom_metadata', {}))
                }
                flattened_memories.append(flat_memory)

            df = pd.DataFrame(flattened_memories)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()

            return StreamingResponse(
                io.StringIO(csv_content),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={agent.config.name}_memories.csv"}
            )

        elif format.lower() == "zip":
            # Create ZIP file with JSON + CSV
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add JSON file
                json_content = json.dumps(export_data, indent=2, ensure_ascii=False)
                zip_file.writestr(f"{agent.config.name}_memories.json", json_content)

                # Add CSV file if memories exist
                if all_memories:
                    try:
                        import pandas as pd
                        flattened_memories = []
                        for memory in all_memories:
                            flat_memory = {
                                'id': memory.get('id'),
                                'content': memory.get('content'),
                                'memory_type': memory.get('memory_type'),
                                'emotion_score': memory.get('emotion_score', 0.0),
                                'coherence_score': memory.get('coherence_score', 0.5),
                                'novelty_score': memory.get('novelty_score', 0.5),
                                'salience': memory.get('salience', 0.5),
                                'created_at': memory.get('created_at'),
                                'accessed_count': memory.get('accessed_count', 0),
                                'last_accessed': memory.get('last_accessed'),
                                'custom_metadata': json.dumps(memory.get('custom_metadata', {}))
                            }
                            flattened_memories.append(flat_memory)

                        df = pd.DataFrame(flattened_memories)
                        csv_content = df.to_csv(index=False)
                        zip_file.writestr(f"{agent.config.name}_memories.csv", csv_content)
                    except ImportError:
                        logger.warning("pandas not available for CSV export in ZIP")

                # Add metadata file
                metadata = {
                    "agent_id": agent_id,
                    "agent_name": agent.config.name,
                    "persona": agent.config.persona,
                    "detailed_persona": agent.config.detailed_persona,
                    "export_timestamp": datetime.now().isoformat(),
                    "total_memories": len(all_memories),
                    "memory_types": list(agent.memory_blossom.memory_stores.keys())
                }
                zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))

            zip_buffer.seek(0)

            return StreamingResponse(
                io.BytesIO(zip_buffer.read()),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={agent.config.name}_memories.zip"}
            )

        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'json', 'csv', or 'zip'")

    except Exception as e:
        logger.error(f"Error exporting memories for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# ==============================================================================
# ==================== Mind Visualization Endpoints =======================
# ==============================================================================

@app.get("/agents/{agent_id}/recent-memories", tags=["Agent Visualization"])
async def get_agent_recent_memories(agent_id: str, limit: int = 3, current_user: dict = Depends(verify_token)):
    """
    Retrieves the most recently created memories for an agent.
    Works for both NCF and CEAF agents via the adapter interface.
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership or public status
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied. You can only view memories of your own agents.")

    try:
        all_memories_raw = agent.memory_blossom.get_all_memories()

        # --- START FIX: Standardize all items to dictionaries before processing ---
        all_memories_as_dicts = []
        for mem in all_memories_raw:
            if hasattr(mem, 'to_dict') and callable(mem.to_dict):
                all_memories_as_dicts.append(mem.to_dict())  # Convert NCF Memory object
            elif isinstance(mem, dict):
                all_memories_as_dicts.append(mem)  # Already a dict (from CEAF adapter)

        # Sort by timestamp using the now-consistent dictionary format
        def get_timestamp(mem_dict):
            # NCF .to_dict() uses 'creation_time', CEAF adapter uses 'timestamp'
            ts_str = mem_dict.get('creation_time') or mem_dict.get('timestamp')
            return datetime.fromisoformat(ts_str) if ts_str else datetime.min

        sorted_memories = sorted(all_memories_as_dicts, key=get_timestamp, reverse=True)

        return sorted_memories[:limit]
        # --- END FIX ---

    except Exception as e:
        logger.error(f"Error fetching recent memories for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve recent memories.")


@app.get("/agents/{agent_id}/ceaf-status", tags=["Agent Visualization", "CEAF"])
async def get_ceaf_agent_status(agent_id: str, current_user: dict = Depends(verify_token)):
    """
    Retrieves the specific internal status (MCL, NCIM) for a CEAF agent.
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership or public status
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied. You can only view the status of your own agents.")

    # This endpoint is strictly for CEAF agents
    if not isinstance(agent, CEAFAgentAdapter):
        raise HTTPException(status_code=400, detail="This endpoint is only for CEAF agents.")

    try:
        # get_enhanced_stats on the adapter is the perfect place to get this info
        stats = agent.get_enhanced_stats()
        ceaf_stats = stats.get('ceaf_stats', {})

        # Extract the specific data needed for the UI widgets
        mcl_state = ceaf_stats.get('mcl_stats', {}).get('current_state', 'Unknown')
        identity_history = agent.ceaf_system.ncim.get_identity_history()

        original_identity = identity_history[0]['narrative'] if identity_history else "Not set."
        current_identity = identity_history[-1]['narrative'] if identity_history else "Not set."

        return {
            "agent_id": agent_id,
            "system_type": "ceaf",
            "mcl_state": mcl_state,
            "identity_evolution": {
                "original": original_identity,
                "current": current_identity
            }
        }

    except Exception as e:
        logger.error(f"Error fetching CEAF status for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve CEAF status.")


# ==================== END OF Mind Visualization Endpoints =======================


@app.post("/agents/{agent_id}/memories/upload", response_model=BulkMemoryUploadResponse)
async def upload_agent_memories(
        agent_id: str,
        request: MemoryUploadRequest,
        current_user: dict = Depends(verify_token)
):
    """
    Upload new memories to a specific agent
    Supports bulk upload with validation and error handling
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        successful_uploads = 0
        failed_uploads = 0
        errors = []
        memory_types_added = set()

        for i, memory_data in enumerate(request.memories):
            try:
                # Validate required fields
                if request.validate_format:
                    required_fields = ['content', 'memory_type']
                    missing_fields = [field for field in required_fields if field not in memory_data]
                    if missing_fields:
                        error_msg = f"Memory {i}: Missing required fields: {missing_fields}"
                        errors.append(error_msg)
                        failed_uploads += 1
                        continue

                # Set default values for optional fields
                memory_content = memory_data['content']
                memory_type = memory_data['memory_type']
                emotion_score = float(memory_data.get('emotion_score', 0.0))
                initial_salience = float(memory_data.get('initial_salience', 0.5))
                custom_metadata = memory_data.get('custom_metadata', {})

                # Add upload metadata
                custom_metadata.update({
                    "source": "user_upload",
                    "uploaded_by": current_user["user_id"],
                    "upload_timestamp": datetime.now().isoformat()
                })

                # Check for existing memory if not overwriting
                if not request.overwrite_existing:
                    existing_memories = agent.memory_blossom.memory_stores.get(memory_type, [])
                    content_exists = any(mem.content == memory_content for mem in existing_memories)
                    if content_exists:
                        error_msg = f"Memory {i}: Content already exists (use overwrite_existing=true to force)"
                        errors.append(error_msg)
                        failed_uploads += 1
                        continue

                # Add memory to agent's isolated memory system
                agent.memory_blossom.add_memory(
                    content=memory_content,
                    memory_type=memory_type,
                    emotion_score=emotion_score,
                    initial_salience=initial_salience,
                    custom_metadata=custom_metadata
                )

                memory_types_added.add(memory_type)
                successful_uploads += 1

            except Exception as e:
                error_msg = f"Memory {i}: Failed to upload - {str(e)}"
                errors.append(error_msg)
                failed_uploads += 1
                logger.error(f"Error uploading memory {i}: {e}")

        # Save all memories to disk
        if successful_uploads > 0:
            agent.memory_blossom.save_memories()

        return BulkMemoryUploadResponse(
            agent_id=agent_id,
            agent_name=agent.config.name,
            upload_timestamp=datetime.now().isoformat(),
            total_uploaded=len(request.memories),
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads,
            errors=errors,
            memory_types_added=list(memory_types_added)
        )

    except Exception as e:
        logger.error(f"Error uploading memories for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/agents/{agent_id}/memories/upload/file")
async def upload_memories_from_file(
        agent_id: str,
        file: UploadFile = File(...),
        overwrite_existing: bool = False,
        current_user: dict = Depends(verify_token)
):
    """
    Upload memories from a JSON or CSV file
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # Read file content
        content = await file.read()

        if file.filename.endswith('.json'):
            # Parse JSON file
            try:
                data = json.loads(content.decode('utf-8'))

                # Handle different JSON structures
                if isinstance(data, dict) and 'memories' in data:
                    memories = data['memories']  # Export format
                elif isinstance(data, list):
                    memories = data  # Direct list format
                else:
                    raise HTTPException(status_code=400, detail="Invalid JSON format")

            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

        elif file.filename.endswith('.csv'):
            # Parse CSV file
            try:
                import pandas as pd
            except ImportError:
                raise HTTPException(status_code=500, detail="pandas not installed. Please install: pip install pandas")

            try:
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                memories = []

                for _, row in df.iterrows():
                    memory = {
                        'content': row.get('content'),
                        'memory_type': row.get('memory_type'),
                        'emotion_score': float(row.get('emotion_score', 0.0)),
                        'initial_salience': float(row.get('salience', 0.5)),
                    }

                    # Parse custom_metadata if it exists
                    if 'custom_metadata' in row and pd.notna(row['custom_metadata']):
                        try:
                            memory['custom_metadata'] = json.loads(row['custom_metadata'])
                        except:
                            memory['custom_metadata'] = {}

                    memories.append(memory)

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use .json or .csv")

        # Create upload request
        upload_request = MemoryUploadRequest(
            memories=memories,
            overwrite_existing=overwrite_existing,
            validate_format=True
        )

        # Use existing upload logic
        return await upload_agent_memories(agent_id, upload_request, current_user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


# --- Memory Analytics ---
@app.get("/agents/{agent_id}/memories/analytics")
async def get_memory_analytics(
        agent_id: str,
        current_user: dict = Depends(verify_token)
):
    """
    Get detailed analytics about agent's memory system
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        analytics = {
            "agent_id": agent_id,
            "agent_name": agent.config.name,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_memories": 0,
            "memory_types": {},
            "emotion_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "salience_distribution": {"high": 0, "medium": 0, "low": 0},
            "recent_activity": {"last_7_days": 0, "last_30_days": 0},
            "top_memory_sources": {},
            "memory_timeline": []
        }

        all_memories = []

        # Collect all memories
        for mem_type, mem_list in agent.memory_blossom.memory_stores.items():
            analytics["memory_types"][mem_type] = len(mem_list)
            all_memories.extend([(mem, mem_type) for mem in mem_list])

        analytics["total_memories"] = len(all_memories)

        # Analyze memories
        for memory, mem_type in all_memories:
            memory_dict = memory.to_dict()

            # Emotion analysis
            emotion_score = memory_dict.get('emotion_score', 0.0)
            if emotion_score > 0.1:
                analytics["emotion_distribution"]["positive"] += 1
            elif emotion_score < -0.1:
                analytics["emotion_distribution"]["negative"] += 1
            else:
                analytics["emotion_distribution"]["neutral"] += 1

            # Salience analysis
            salience = memory_dict.get('salience', 0.5)
            if salience > 0.7:
                analytics["salience_distribution"]["high"] += 1
            elif salience > 0.3:
                analytics["salience_distribution"]["medium"] += 1
            else:
                analytics["salience_distribution"]["low"] += 1

            # Source analysis
            metadata = memory_dict.get('custom_metadata', {})
            source = metadata.get('source', 'unknown')
            analytics["top_memory_sources"][source] = analytics["top_memory_sources"].get(source, 0) + 1

            # Timeline analysis (if created_at exists)
            created_at = memory_dict.get('created_at')
            if created_at:
                try:
                    if isinstance(created_at, str):
                        created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        created_date = created_at

                    days_ago = (datetime.now() - created_date.replace(tzinfo=None)).days

                    if days_ago <= 7:
                        analytics["recent_activity"]["last_7_days"] += 1
                    if days_ago <= 30:
                        analytics["recent_activity"]["last_30_days"] += 1

                except:
                    pass

        return analytics

    except Exception as e:
        logger.error(f"Error generating memory analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")


# --- Helper Endpoints ---
@app.get("/agents/{agent_id}/memory-types")
async def get_agent_memory_types(
        agent_id: str,
        current_user: dict = Depends(verify_token)
):
    """
    Get list of memory types available for this agent
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    memory_types = {}
    for mem_type, mem_list in agent.memory_blossom.memory_stores.items():
        memory_types[mem_type] = {
            "count": len(mem_list),
            "description": f"Memories of type '{mem_type}'"
        }

    return {
        "agent_id": agent_id,
        "agent_name": agent.config.name,
        "memory_types": memory_types,
        "total_types": len(memory_types)
    }


# --- NCF Information Routes ---
@app.get("/agents/{agent_id}/ncf-status")
async def get_ncf_status(agent_id: str, current_user: dict = Depends(verify_token)):
    """Get NCF capabilities status for a specific agent"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Check access permissions
    if agent.config.user_id != current_user["user_id"]:
        db_agent = db_repo.get_agent(agent_id)
        if not db_agent or not db_agent.is_public:
            raise HTTPException(status_code=403, detail="Access denied")

    return {
        "agent_id": agent_id,
        "agent_name": agent.config.name,
        "ncf_enabled": True,
        "capabilities": {
            "narrative_foundation": True,
            "rag_retrieval": True,
            "reflector_analysis": True,
            "isolated_memory_system": True,
            "contextual_prompting": True,
            "conversation_history_tracking": True,
            "memory_types_supported": ["Explicit", "Emotional", "Procedural", "Flashbulb", "Liminal", "Generative"]
        },
        "memory_statistics": {
            "total_memories": sum(len(m) for m in agent.memory_blossom.memory_stores.values()),
            "memory_stores": {mem_type: len(mem_list) for mem_type, mem_list in
                              agent.memory_blossom.memory_stores.items()},
            "memory_persistence_path": agent.config.memory_path
        },
        "model_info": {
            "model": agent.config.model,
            "llm_instance": "LiteLlm"
        }
    }


# --- Pre-built Agent Helpers and Admin Routes ---
def get_archetype_description(archetype: AgentArchetype) -> str:
    """Get the description for an archetype"""
    descriptions = {
        AgentArchetype.PHILOSOPHER: "Deep thinker who questions everything and seeks wisdom through inquiry",
        AgentArchetype.CREATIVE: "Imaginative soul who connects ideas and sees possibilities everywhere",
        AgentArchetype.THERAPIST: "Compassionate listener who creates safe spaces for authentic expression",
        AgentArchetype.SCIENTIST: "Methodical researcher with systematic approach to understanding",
        AgentArchetype.TEACHER: "Patient educator who helps others learn and grow",
        AgentArchetype.REBEL: "Challenger of status quo who questions authority and conventions",
        AgentArchetype.SAGE: "Wise counselor with balanced perspective and deep understanding",
        AgentArchetype.EXPLORER: "Curious adventurer always seeking new experiences and knowledge",
        AgentArchetype.GUARDIAN: "Protective presence who provides stability and security",
        AgentArchetype.TRICKSTER: "Playful spirit who uses humor and irony to reveal truths"
    }
    return descriptions.get(archetype, "Unique personality with distinct traits")


async def process_agent_training(
        agent_id: str,
        conversations: List[TrainingConversation],
        target_traits: Optional[List[str]]
):
    """Process agent training in the background"""
    try:
        logger.info(f"Starting training for pre-built agent {agent_id} with {len(conversations)} conversations")

        # Simulate training processing
        for i, conv in enumerate(conversations):
            await asyncio.sleep(0.1)  # Simulate analysis
            logger.info(f"Processed conversation {i + 1}/{len(conversations)}")

        # Update agent stats
        agent = prebuilt_repo.agents.get(agent_id)
        if not agent:
            logger.error(f"Training failed: Pre-built agent {agent_id} not found.")
            return

        agent.total_interactions += len(conversations)
        agent.last_training_date = datetime.now()

        # Calculate new rating based on feedback
        ratings = [conv.rating for conv in conversations]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            # Update agent rating (weighted average)
            agent.rating = (agent.rating * 0.8) + (avg_rating * 0.2)

        # Save the updated agent
        agent_path = prebuilt_repo.storage_path / f"{agent_id}.json"
        with open(agent_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create new memories from the training
        new_memories = []
        for conv in conversations:
            memory = {
                "content": f"Training conversation: User said '{conv.user_message}', I responded '{conv.agent_response}'. Rating: {conv.rating}/5",
                "memory_type": "Procedural",
                "emotion_score": (conv.rating - 3) / 2,  # Convert 1-5 to -1 to 1
                "initial_salience": conv.rating / 5,
                "custom_metadata": {
                    "source": "training",
                    "rating": conv.rating,
                    "notes": conv.notes,
                    "trained_at": datetime.now().isoformat()
                }
            }
            new_memories.append(memory)

        if 'memories' not in data:
            data['memories'] = []
        data["memories"].extend(new_memories)

        # Update agent data in the file
        data["agent"]["total_interactions"] = agent.total_interactions
        data["agent"]["rating"] = agent.rating
        data["agent"]["last_training_date"] = agent.last_training_date.isoformat()

        with open(agent_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Completed training for agent {agent_id}")

    except Exception as e:
        logger.error(f"Error training agent {agent_id} in background: {e}", exc_info=True)


@app.post("/admin/prebuilt-agents/create")
async def create_prebuilt_agent_admin(
        request: CreatePrebuiltRequest,
        current_user: dict = Depends(verify_token)
):
    """Create a new pre-built agent (admin only)"""
    # Adjust admin logic as needed for your application
    if current_user["username"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        archetype = AgentArchetype(request.archetype)

        agent = prebuilt_repo.create_prebuilt_agent(
            name=request.name,
            archetype=archetype,
            system_type=request.system_type,
            custom_traits=request.custom_traits
        )

        return {
            "agent_id": agent.id,
            "message": f"Created prebuilt agent '{agent.name}'",
            "archetype": agent.archetype.value,
            "system_type": agent.system_type
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {e}")
    except Exception as e:
        logger.error(f"Error creating prebuilt agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create prebuilt agent")


@app.post("/admin/prebuilt-agents/train")
async def train_prebuilt_agent_admin(
        request: TrainAgentRequest,
        background_tasks: BackgroundTasks,
        current_user: dict = Depends(verify_token)
):
    """Train a pre-built agent with conversations (admin only)"""
    if current_user["username"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    if request.agent_id not in prebuilt_repo.agents:
        raise HTTPException(status_code=404, detail="Prebuilt agent not found")

    # Process training in the background
    background_tasks.add_task(
        process_agent_training,
        request.agent_id,
        request.conversations,
        request.target_traits
    )

    return {
        "message": f"Started training for agent {request.agent_id}",
        "training_conversations": len(request.conversations),
        "status": "processing_in_background"
    }


# --- Frontend Routes ---
# Este bloco deve vir DEPOIS de todas as suas rotas de API

# Define o caminho para a pasta frontend de forma robusta
frontend_path = Path(__file__).resolve().parent.parent / "frontend"


# A rota principal ("/") que lida com tudo.
# Ela tentará servir o index.html, e se não encontrar, servirá a mensagem JSON.
@app.get("/")
async def serve_frontend_or_api_info():
    index_path = frontend_path / "index.html"

    if index_path.is_file():  # Verifica se o arquivo existe e é um arquivo
        return FileResponse(index_path)
    else:
        # Este é o "fallback" que só é executado se o frontend não for encontrado
        return {
            "message": "Aura Multi-Agent API (NCF-Enabled) is running. Frontend not found.",
            "status_check": f"Checked for frontend at: {index_path}",
            "version": "2.1.0",
            "endpoints_docs": "/docs"
        }


# Monta um diretório "estático" para servir outros arquivos do frontend (CSS, JS, imagens)
# Se você tiver esses arquivos, eles devem estar na pasta 'frontend'
# e ser referenciados no HTML como, por exemplo, "/static/css/styles.css"
if frontend_path.is_dir():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

app.mount("/agent_files", StaticFiles(directory=AGENT_STORAGE_PATH), name="agent_files")


@app.get("/upload")
async def serve_upload_page():
    """Serve the agent creation/upload page"""
    upload_path = frontend_path / "upload.html"

    if upload_path.is_file():
        return FileResponse(upload_path)
    else:
        raise HTTPException(status_code=404, detail="Upload page not found")


# --- Health Check ---
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "running",
            "database": "connected",
            "agent_manager": "initialized_with_ncf",
            "prebuilt_agent_repo": "initialized",
            "ncf_capabilities": "enabled"
        },
        "version": "2.1.0",
        "ncf_enabled": True,
        "enhanced_features": [
            "memory_export_import",
            "agent_profile_editing",
            "memory_analytics",
            "bulk_memory_operations",
            "prebuilt_agents"
        ]
    }


# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "ncf_enabled": True
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "ncf_enabled": True
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
