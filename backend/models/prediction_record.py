
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Integer,
    Float,
    String,
    ForeignKey,
    DateTime,
    CheckConstraint
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship
)

from backend.database import Base


# 只供 Pylance 等类型检查工具使用
if TYPE_CHECKING:
    from backend.models.review import Review


class PredictionRecord(Base):

    __tablename__ = "prediction_records"

    # 1. 数据库约束
    __table_args__ = (
        CheckConstraint(
            "predicted_label IN (0, 1)",
            name="ck_prediction_label"
        ),
        CheckConstraint(
            "helpful_probability >= 0 "
            "AND helpful_probability <= 1",
            name="ck_helpful_probability"
        ),
    )

    # 2. 预测记录 ID（主键）
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True
    )

    # 3. 被预测的评论 ID（外键）
    review_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("reviews.review_id"),
        nullable=False,
        index=True
    )

    # 4. 预测标签
    # 0：没有有用票
    # 1：有至少一张有用票
    predicted_label: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    # 5. 预测为正类的概率
    helpful_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    # 6. 模型版本
    model_version: Mapped[str] = mapped_column(
        String(64),
        default="graphsage-v1",
        nullable=False
    )

    # 7. 预测时间（UTC）
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(
            tzinfo=None
        ),
        nullable=False
    )

    # 8. ORM 关联：预测记录属于一条评论
    review: Mapped["Review"] = relationship(
        "Review",
        back_populates="predictions"
    )
