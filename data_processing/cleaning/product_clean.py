
import json
from pathlib import Path

# 1. 文件路径
BASE_DIR = Path(__file__).resolve().parent.parent

META_FILE = BASE_DIR / "data/raw/meta_All_Beauty.jsonl"
REVIEWS_FILE = BASE_DIR / "data/processed/clean_reviews.jsonl"
OUTPUT_FILE = BASE_DIR / "data/processed/clean_products.jsonl"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# 2. 读取清洗后评论中的商品 ID
product_ids = set()

with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        review = json.loads(line)
        product_id = review.get("parent_asin")

        if product_id:
            product_ids.add(product_id)

print("评论涉及的商品数：", len(product_ids))


# 3. 清洗文本字段
def clean_text(value):
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        return " ".join(
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        )

    return ""


# 4. 提取图片链接
def extract_images(images):
    urls = []

    if not isinstance(images, list):
        return urls

    for image in images:
        if not isinstance(image, dict):
            continue

        url = (
            image.get("large")
            or image.get("hi_res")
            or image.get("thumb")
        )

        if isinstance(url, str) and url.startswith(
            ("http://", "https://")
        ):
            if url not in urls:
                urls.append(url)

    return urls


# 5. 读取并清洗商品元数据
total = 0
matched = 0
no_image = 0
duplicates = 0

seen_products = set()

with open(META_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        total += 1
        product = json.loads(line)

        product_id = product.get("parent_asin")

        # 只保留评论数据中出现的商品
        if product_id not in product_ids:
            continue

        # 去重
        if product_id in seen_products:
            duplicates += 1
            continue

        seen_products.add(product_id)

        # 商品文本
        title = clean_text(product.get("title"))
        description = clean_text(
            product.get("description")
        )
        features = clean_text(product.get("features"))

        # 商品图片
        images = extract_images(product.get("images", []))

        if not images:
            no_image += 1

        # 生成清洗后的商品记录
        clean_product = {
            "parent_asin": product_id,
            "title": title,
            "description": description,
            "features": features,
            "store": clean_text(product.get("store")),
            "image_urls": images,
            "has_image": len(images) > 0
        }

        fout.write(
            json.dumps(clean_product, ensure_ascii=False) + "\n"
        )

        matched += 1

print("\n商品元数据清洗完成")
print("原始商品记录数：", total)
print("评论涉及商品数：", len(product_ids))
print("匹配商品数：", matched)
print("未匹配商品数：", len(product_ids) - matched)
print("重复商品记录数：", duplicates)
print("没有图片的商品数：", no_image)
print("有图片的商品数：", matched - no_image)
