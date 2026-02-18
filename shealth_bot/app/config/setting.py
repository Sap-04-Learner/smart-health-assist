from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from functools import lru_cache


# =================== Custom Exceptions ===================


class HealthAssistantError(Exception):
    """Base exception for Health Assistant application."""
    pass


class DocumentParsingError(HealthAssistantError):
    """Exception raised during document parsing."""
    pass


class EmbeddingError(HealthAssistantError):
    """Exception raised during embedding generation."""
    pass


class SessionError(HealthAssistantError):
    """Exception raised during session management."""
    pass


# =================== Settings ===================


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Ollama settings
    ollama_host: str = 'http://localhost:11434'
    ollama_model: str = 'mistral'

    # Embedding model settings
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'

    # ChromaDB settings
    chroma_db_path: str = './data/chroma_db'
    collection_name: str = 'health_knowledge'

    # Server settings
    server_host: str = 'localhost'
    server_port: int = 8000  

    # Device settings
    embedding_device_populate: str = 'auto'  # GPU for bulk population
    embedding_device_runtime: str = 'cpu'    # CPU for runtime queries

    # CORS settings
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:80",
        "https://rhett-subnasal-wholly.ngrok-free.dev"  
    ]
    allowed_methods: list[str] = ["GET", "POST", "OPTIONS"]
    allowed_headers: list[str] = ["Content-Type", "Authorization"]

    # Rate limiting settings
    rate_limit_per_minute: int = 30

    # ChromaDB batch settings
    chroma_batch_size: int = 5000

    # Logging & paths
    log_level: str = "INFO"
    app_version: str = "3.0"
    session_storage_path: str = "./data/sessions"
    documents_path: str = "./data/documents"
    logs_path: str = "./logs"

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'  # Ignore extra fields in .env
    )


# =================== Cached Settings Instance ===================


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    Useful for dependency injection (FastAPI) and testing.
    """
    return Settings()


# Instantiate once for simple imports (still works as before)
settings = get_settings()


# =================== Directory Setup Helper ===================


def ensure_directories(settings: Settings):
    """
    Create required directories based on settings.
    Call this once at application startup.
    """
    os.makedirs(settings.logs_path, exist_ok=True)
    os.makedirs(settings.session_storage_path, exist_ok=True)
    os.makedirs(settings.documents_path, exist_ok=True)
    os.makedirs(settings.chroma_db_path, exist_ok=True)