from backend.config import settings
from backend.database import check_database_connection


def main():

    print("数据库主机：", settings.DB_HOST)
    print("数据库端口：", settings.DB_PORT)
    print("数据库名称：", settings.DB_NAME)
    print("数据库用户：", settings.DB_USER)

    print("\n开始测试数据库连接...")

    success = check_database_connection()

    if success:
        print("MySQL 数据库连接成功！")
    else:
        print("MySQL 数据库连接失败！")


if __name__ == "__main__":
    main()