
from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator
)


# ========== 1. 已有评论预测请求 ==========

class PredictionByIdRequest(BaseModel):

    review_id: str = Field(
        min_length=1,
        max_length=64,
        description="已有评论 ID"
    )


# ========== 2. 新评论预测请求 ==========

class PredictionRequest(BaseModel):

    parent_asin: str = Field(
        min_length=1,
        max_length=32,
        description="所属商品 ID"
    )

    title: str | None = Field(
        default=None,
        max_length=500,
        description="评论标题"
    )

    text: str = Field(
        min_length=1,
        description="评论正文"
    )

    rating: float = Field(
        ge=1,
        le=5,
        description="商品评分"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:

        value = value.strip()

        if not value:
            raise ValueError("评论正文不能为空")

        return value


# ========== 3. 预测结果响应 ==========

class PredictionResponse(BaseModel):

    review_id: str

    predicted_label: int = Field(
        ge=0,
        le=1,
        description="0=无有用票，1=至少一张有用票"
    )

    helpful_probability: float = Field(
        ge=0,
        le=1,
        description="预测正类的概率"
    )

    model_version: str

    model_config = ConfigDict(
        from_attributes=True
    )


# ========== 4. 历史预测记录 ==========

class PredictionRecordResponse(PredictionResponse):

    id: int

    created_at: datetime


# ========== 5. 分页响应 ==========

class PredictionHistoryResponse(BaseModel):

    total: int = Field(ge=0)

    page: int = Field(ge=1)

    page_size: int = Field(ge=1)

    items: list[PredictionRecordResponse]
