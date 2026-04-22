from pydantic_settings import BaseSettings, SettingsConfigDict

# Clase Settings
class Settings(BaseSettings):
    GROQ_API_KEY: str
    CHROMA_PERSIST_DIR: str = './storage/chroma'
    UPLOAD_DIR: str = './storage/uploads'
    MAX_PDF_SIZE_MB: int = 50
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(
        env_file=".env",            # Indica el archivo donde leer
        env_file_encoding="utf-8",
        case_sensitive=False
    )

settings = Settings()