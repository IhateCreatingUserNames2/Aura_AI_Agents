# Aura/init_db.py
from database.models import Base, User, AgentRepository
from sqlalchemy import create_engine
import bcrypt
import os

def init_database():
    db_url = os.getenv("DATABASE_URL", "sqlite:///aura_agents.db")
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    print(f"Database initialized at: {db_url}")

def create_demo_user():
    repo = AgentRepository()
    # Create a demo user (implement user creation in repository)
    # This is just an example
    print("Database ready for users!")

if __name__ == "__main__":
    init_database()
    create_demo_user()