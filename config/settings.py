from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
import os

# Load variables from .env file
load_dotenv()

@dataclass
class Settings:
    """
    Central configuration class.
    All environment variables are loaded once
    and accessed across the project.
    """

    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # Vector DB settings
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH")

    # Embedding settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")

    # Chunk settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP"))

    # Retrieval settings
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS"))

    # LLM settings
    GPT_MODEL_NAME: str = os.getenv("GPT_MODEL_NAME")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

    # Generation settings
    TEMPERATURE: float = float(os.getenv("TEMPERATURE"))

    # Web search settings
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")
    TOP_K_WEB_RESULTS: int = int(os.getenv("TOP_K_WEB_RESULTS"))

settings = Settings()
