
from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator
)


# ========== 1. 公共字段 ==========

class ReviewBase(BaseModel):

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
        description="商品评分，范围 1~5"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:

        value = value.strip()

        if not value:
            raise ValueError("评论内容不能为空")

        return value


# ========== 2. 创建评论 ==========

class ReviewCreate(ReviewBase):

    # 可由前端提供，也可以由后端生成
    review_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64
    )

    review_timestamp: int | None = Field(
        default=None,
        ge=0,
        description="原始评论时间戳，毫秒"
    )


# ========== 3. 更新评论 ==========

class ReviewUpdate(BaseModel):

    title: str | None = Field(
        default=None,
        max_length=500
    )

    text: str | None = None

    rating: float | None = Field(
        default=None,
        ge=1,
        le=5
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str | None):

        if value is None:
            return value

        value = value.strip()

        if not value:
            raise ValueError("评论内容不能为空")

        return value


# ========== 4. 评论响应 ==========

class ReviewResponse(ReviewBase):

    review_id: str

    helpful_vote: int | None = None
    label: int | None = None

    review_timestamp: int | None = None
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )
