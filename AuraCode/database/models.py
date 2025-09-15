# database/models.py
from sqlalchemy import create_engine, Column, String, DateTime, JSON, ForeignKey, Text, Float, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import uuid
from typing import Optional

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    agents = relationship("Agent", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("ChatSession", back_populates="user")
    # Credits
    credits = Column(Integer, default=1000, nullable=False)  # Start new users with 1000 credits



# =================== TABELA live memories ====================
class LiveMemory(Base):
    __tablename__ = 'live_memories'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=False)
    memory_type = Column(String, nullable=False)
    custom_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_indexed = Column(Integer, default=0, nullable=False) # 0 = não indexado, 1 = indexado

class Agent(Base):
    __tablename__ = 'agents'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    name = Column(String, nullable=False)
    persona = Column(String, nullable=False)
    detailed_persona = Column(Text)
    avatar_url = Column(String, nullable=True)
    model = Column(String, default="openrouter/openai/gpt-4o-mini")
    settings = Column(JSON, default=dict)
    is_public = Column(Integer, default=0)  # 0 = private, 1 = public

    # ---  CAMPOS PARA VERSIONAMENTO E MARKETPLACE ---
    is_public_template = Column(Integer, default=0, nullable=False) # 1 se for um template no marketplace
    parent_agent_id = Column(String, ForeignKey('agents.id'), nullable=True) # Para agrupar versões
    version = Column(String, default="1.0.0", nullable=False)
    changelog = Column(Text, nullable=True) # Notas da versão
    clone_count = Column(Integer, default=0, nullable=False)
    # ---  FIM DE CAMPOS PARA VERSIONAMENTO E MARKETPLACE ---

    usage_cost = Column(Float, default=0.0, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Memory configuration
    memory_temperature = Column(Float, default=0.7)
    coherence_bias = Column(Float, default=0.6)
    novelty_bias = Column(Float, default=0.4)

    # Relationships
    user = relationship("User", back_populates="agents")
    # --- Relacionamento para versões ---
    versions = relationship("Agent", back_populates="parent", remote_side=[id])
    parent = relationship("Agent", remote_side=[parent_agent_id])
    # --- FIM DE  Relacionamento para versões ---
    sessions = relationship("ChatSession", back_populates="agent", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="agent", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = 'sessions'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    agent_id = Column(String, ForeignKey('agents.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="sessions")
    agent = relationship("Agent", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = 'messages'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey('sessions.id'), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class Memory(Base): # This is database.models.Memory
    __tablename__ = 'memories'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey('agents.id'), nullable=False)
    content = Column(Text, nullable=False)
    memory_type = Column(String, nullable=False)
    custom_metadata = Column(JSON, default=dict) # RENAMED from metadata

    # Scores
    emotion_score = Column(Float, default=0.0)
    coherence_score = Column(Float, default=0.5)
    novelty_score = Column(Float, default=0.5)
    salience = Column(Float, default=0.5)
    decay_factor = Column(Float, default=1.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)

    # Embedding stored as JSON array
    embedding = Column(JSON)

    # Relationships
    agent = relationship("Agent", back_populates="memories")
    connections = relationship("MemoryConnection",
                               foreign_keys="MemoryConnection.source_memory_id",
                               back_populates="source_memory",
                               cascade="all, delete-orphan")


class MemoryConnection(Base):
    __tablename__ = 'memory_connections'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_memory_id = Column(String, ForeignKey('memories.id'), nullable=False)
    target_memory_id = Column(String, ForeignKey('memories.id'), nullable=False)
    strength = Column(Float, default=0.5)
    relation_type = Column(String)

    # Relationships
    source_memory = relationship("Memory", foreign_keys=[source_memory_id])
    target_memory = relationship("Memory", foreign_keys=[target_memory_id])


class CreditTransaction(Base):
    __tablename__ = 'credit_transactions'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    agent_id = Column(String, nullable=True)
    amount = Column(Integer, nullable=False)  # Can be positive (add) or negative (spend)
    model_used = Column(String, nullable=True)
    description = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")

# Database Repository
class AgentRepository:
    def __init__(self, db_url: str = "sqlite:///aura_agents.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Fetches a user by their unique ID."""
        with self.SessionLocal() as session:
            return session.query(User).filter(User.id == user_id).first()

    def create_agent(self, agent_id: str, user_id: str, name: str, persona: str,
                     detailed_persona: str, model: str = None, is_public: bool = False) -> Agent:
        with self.SessionLocal() as session:
            agent = Agent(
                id=agent_id,
                user_id=user_id,
                name=name,
                persona=persona,
                detailed_persona=detailed_persona,
                model=model or "openrouter/openai/gpt-4o-mini",
                is_public=1 if is_public else 0
            )
            session.add(agent)
            session.commit()
            session.refresh(agent)
            return agent

    def get_agent(self, agent_id: str) -> Agent:
        with self.SessionLocal() as session:
            return session.query(Agent).filter(Agent.id == agent_id).first()

    def list_user_agents(self, user_id: str):
        with self.SessionLocal() as session:
            return session.query(Agent).filter(Agent.user_id == user_id).all()

    def delete_agent(self, agent_id: str, user_id: str) -> bool:
        with self.SessionLocal() as session:
            agent = session.query(Agent).filter(
                Agent.id == agent_id,
                Agent.user_id == user_id
            ).first()
            if agent:
                session.delete(agent)
                session.commit()
                return True
            return False

        # --- FUNÇÕES PARA LiveMemory ---

    def save_live_memory(self, content: str, memory_type: str, metadata: dict) -> LiveMemory:
        with self.SessionLocal() as session:
            memory = LiveMemory(
                content=content,
                memory_type=memory_type,
                custom_metadata=metadata,
                is_indexed=0  # Sempre começa como não indexado
            )
            session.add(memory)
            session.commit()
            session.refresh(memory)
            return memory

    def get_unindexed_live_memories(self, limit: int = 100):
        with self.SessionLocal() as session:
            return session.query(LiveMemory).filter(LiveMemory.is_indexed == 0).limit(limit).all()

    def mark_live_memories_as_indexed(self, memory_ids: list[str]):
        if not memory_ids:
            return
        with self.SessionLocal() as session:
            session.query(LiveMemory).filter(LiveMemory.id.in_(memory_ids)).update({"is_indexed": 1},
                                                                                   synchronize_session=False)
            session.commit()


    def save_memory(self, agent_id: str, memory_data: dict): # memory_data is from memory_system.memory_models.Memory.to_dict()
        with self.SessionLocal() as session:
            memory = Memory( # This is database.models.Memory
                agent_id=agent_id,
                content=memory_data['content'],
                memory_type=memory_data['memory_type'],
                custom_metadata=memory_data.get('custom_metadata', {}), # UPDATED KEY
                emotion_score=memory_data.get('emotion_score', 0.0),
                coherence_score=memory_data.get('coherence_score', 0.5),
                novelty_score=memory_data.get('novelty_score', 0.5),
                salience=memory_data.get('salience', 0.5), # Ensure this aligns with memory_system.memory_models.Memory
                embedding=memory_data.get('embedding', []) # Ensure this aligns
            )
            session.add(memory)
            session.commit()
            return memory

    def get_agent_memories(self, agent_id: str, memory_type: str = None, limit: int = 100):
        with self.SessionLocal() as session:
            query = session.query(Memory).filter(Memory.agent_id == agent_id)
            if memory_type:
                query = query.filter(Memory.memory_type == memory_type)
            return query.order_by(Memory.created_at.desc()).limit(limit).all()

    def save_message(self, session_id: str, role: str, content: str):
        with self.SessionLocal() as session:
            message = Message(
                session_id=session_id,
                role=role,
                content=content
            )
            session.add(message)
            session.commit()
            return message

    def get_session_messages(self, session_id: str, limit: int = 50):
        with self.SessionLocal() as session:
            return session.query(Message).filter(
                Message.session_id == session_id
            ).order_by(Message.created_at.desc()).limit(limit).all()