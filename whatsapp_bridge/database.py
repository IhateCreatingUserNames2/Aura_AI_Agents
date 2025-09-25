# whatsapp_bridge/database.py
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./whatsapp_bridge.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class WhatsAppUser(Base):
    __tablename__ = "whatsapp_users"
    phone_number = Column(String, primary_key=True, index=True)
    aura_user_id = Column(String, unique=True, nullable=False)
    aura_auth_token = Column(String, nullable=False)
    selected_agent_id = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)