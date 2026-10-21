
from sqlalchemy import inspect

from backend.database import Base, engine
import backend.models  # 注册全部 ORM 模型


def main():

    print("开始初始化数据库...")

    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)

    tables = inspector.get_table_names()

    print("\n数据库初始化完成！")
    print("当前数据表：")

    for table in tables:
        print("-", table)


if __name__ == "__main__":
    main()
