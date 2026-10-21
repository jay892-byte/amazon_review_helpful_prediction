
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 1. 创建 FastAPI 应用
app = FastAPI(
    title="Amazon 评论有用性预测系统",
    description="基于多模态 GraphSAGE 的评论有用性预测 API",
    version="1.0.0"
)

# 2. 配置跨域
# 开发环境允许本地前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# 3. 根接口
@app.get("/")
def root():
    return {
        "message": "Amazon 评论有用性预测系统",
        "version": "1.0.0"
    }


# 4. 健康检查接口
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "amazon-helpful-prediction"
    }
