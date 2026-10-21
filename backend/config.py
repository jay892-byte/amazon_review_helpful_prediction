
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    # ========== 数据库配置 ==========

    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3306
    DB_NAME: str = "amazon_helpful"
    DB_USER: str = "amazon_app"
    DB_PASSWORD: str = ""

    # ========== 模型配置 ==========

    MODEL_PATH: str = "checkpoints/best_model.pth"

    # ========== 环境变量配置 ==========

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # 验证密码是否配置
    @model_validator(mode="after")
    def validate_password(self):
        if not self.DB_PASSWORD:
            raise ValueError(
                "DB_PASSWORD 未配置，请检查 .env 文件"
            )
        return self

    # MySQL 连接地址
    @property
    def DATABASE_URL(self) -> str:

        username = quote_plus(self.DB_USER)
        password = quote_plus(self.DB_PASSWORD)

        return (
            f"mysql+pymysql://{username}:{password}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            "?charset=utf8mb4"
        )

    # 模型文件路径
    @property
    def MODEL_FILE(self) -> Path:

        path = Path(self.MODEL_PATH)

        if not path.is_absolute():
            path = BASE_DIR / path

        return path


settings = Settings()
