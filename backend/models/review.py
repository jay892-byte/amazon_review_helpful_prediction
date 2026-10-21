
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import (
    String,
    Text,
    Float,
    Integer,
    BigInteger,
    ForeignKey,
    DateTime
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship
)

from backend.database import Base


# 只供 Pylance 等类型检查工具使用
if TYPE_CHECKING:
    from backend.models.product import Product
    from backend.models.prediction_record import PredictionRecord


class Review(Base):

    __tablename__ = "reviews"

    # 1. 评论 ID（主键）
    review_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True
    )

    # 2. 所属商品 ID（外键）
    parent_asin: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("products.parent_asin"),
        nullable=False,
        index=True
    )

    # 3. 评论标题
    title: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )

    # 4. 评论正文
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    # 5. 商品评分
    rating: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    # 6. 有用票数
    helpful_vote: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    # 7. 真实二分类标签
    label: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    # 8. 原始评论时间戳（毫秒）
    review_timestamp: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True
    )

    # 9. 记录创建时间（UTC）
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(
            tzinfo=None
        ),
        nullable=False
    )

    # 10. ORM 关联：评论属于一个商品
    product: Mapped["Product"] = relationship(
        "Product",
        back_populates="reviews"
    )

    # 11. ORM 关联：一条评论有多次预测记录
    predictions: Mapped[list["PredictionRecord"]] = relationship(
        "PredictionRecord",
        back_populates="review"
    )
