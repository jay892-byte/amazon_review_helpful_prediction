from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base

if TYPE_CHECKING:
    from backend.models.review import Review


class Product(Base):

    __tablename__ = "products"

    parent_asin: Mapped[str] = mapped_column(
        String(32),
        primary_key=True
    )

    title: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    store: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    image_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True
    )

    reviews: Mapped[list["Review"]] = relationship(
        "Review",
        back_populates="product"
    )