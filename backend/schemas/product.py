
from pydantic import (
    BaseModel,
    ConfigDict,
    Field
)


# ========== 1. 公共字段 ==========

class ProductBase(BaseModel):

    title: str | None = Field(
        default=None,
        max_length=500,
        description="商品标题"
    )

    description: str | None = Field(
        default=None,
        description="商品描述"
    )

    store: str | None = Field(
        default=None,
        max_length=255,
        description="品牌或商家"
    )

    image_path: str | None = Field(
        default=None,
        max_length=1024,
        description="商品图片路径"
    )


# ========== 2. 创建商品 ==========

class ProductCreate(ProductBase):

    parent_asin: str = Field(
        min_length=1,
        max_length=32,
        description="Amazon 商品 ID"
    )


# ========== 3. 更新商品 ==========

class ProductUpdate(ProductBase):
    pass


# ========== 4. 商品响应 ==========

class ProductResponse(ProductBase):

    parent_asin: str

    model_config = ConfigDict(
        from_attributes=True
    )
