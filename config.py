from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str   # postgresql+asyncpg://...  (SQLAlchemy)
    redis_url: str
    jwt_secret: str

    @property
    def sync_database_url(self) -> str:
        # The LangGraph checkpointer uses psycopg, so it needs a plain postgresql:// URL
        return self.database_url.replace("+asyncpg", "")


settings = Settings()