
from pydantic import ValidationError

from backend.schemas.product import ProductCreate
from backend.schemas.review import ReviewCreate
from backend.schemas.prediction import PredictionRequest


def main():

    print("========== 商品 Schema ==========")

    product = ProductCreate(
        parent_asin="B001234567",
        title="Amazon Test Product"
    )

    print(product.model_dump())


    print("\n========== 评论 Schema ==========")

    review = ReviewCreate(
        parent_asin="B001234567",
        text="This product is excellent.",
        rating=5
    )

    print(review.model_dump())


    print("\n========== 预测请求 Schema ==========")

    request = PredictionRequest(
        parent_asin="B001234567",
        text="Very useful product.",
        rating=4
    )

    print(request.model_dump())


    print("\n========== 错误数据测试 ==========")

    try:
        PredictionRequest(
            parent_asin="B001234567",
            text="   ",
            rating=8
        )

    except ValidationError as e:
        print("数据校验成功拦截错误输入")
        print(e)


if __name__ == "__main__":
    main()
