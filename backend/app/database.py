import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "nexus_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "nexus_pass")
DB_NAME = os.getenv("POSTGRES_DB", "nexus_db")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")  # "localhost" car FastAPI tourne hors Docker pour l'instant
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency FastAPI : ouvre une session DB, la ferme apres la requete."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
