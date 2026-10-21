
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from backend.config import settings
from backend.database import Base, engine

# 必须导入所有 ORM 模型，确保注册到 Base.metadata
import backend.models


def init_database():

    print("=" * 50)
    print("Amazon 评论有用性预测系统")
    print("MySQL 数据库初始化")
    print("=" * 50)

    print(f"数据库主机：{settings.DB_HOST}")
    print(f"数据库端口：{settings.DB_PORT}")
    print(f"数据库名称：{settings.DB_NAME}")
    print(f"数据库用户：{settings.DB_USER}")

    try:
        # 1. 测试数据库连接
        print("\n[1/3] 检查数据库连接...")

        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("连接成功，测试结果：", result.scalar())

        # 2. 创建 ORM 数据表
        print("\n[2/3] 创建数据库表...")

        Base.metadata.create_all(bind=engine)

        # 3. 检查表是否创建成功
        print("\n[3/3] 检查数据库表...")

        inspector = inspect(engine)

        existing_tables = set(
            inspector.get_table_names()
        )

        expected_tables = {
            "products",
            "reviews",
            "prediction_records"
        }

        for table in sorted(expected_tables):
            if table in existing_tables:
                print(f"[OK] {table}")
            else:
                print(f"[失败] {table}")

        missing = expected_tables - existing_tables

        if missing:
            raise RuntimeError(
                f"数据库缺少表：{sorted(missing)}"
            )

        print("\n数据库初始化成功！")

    except SQLAlchemyError as e:
        print("\n数据库初始化失败！")
        print("错误类型：", type(e).__name__)
        print("错误详情：", str(e))
        raise

    finally:
        engine.dispose()


if __name__ == "__main__":
    init_database()
