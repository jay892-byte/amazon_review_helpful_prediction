
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings


# ========== 1. 创建数据库引擎 ==========

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=5,
    max_overflow=10,
    echo=False
)


# ========== 2. 创建 Session 工厂 ==========

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False
)


# ========== 3. ORM 模型基类 ==========

class Base(DeclarativeBase):
    pass


# ========== 4. FastAPI 数据库依赖 ==========

def get_db() -> Generator[Session, None, None]:

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# ========== 5. 检查数据库连接 ==========

def check_database_connection() -> bool:

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("数据库查询结果：", result.scalar())

        return True

    except Exception as e:
        print("数据库连接错误：", type(e).__name__)
        print("详细原因：", str(e))
        return False
