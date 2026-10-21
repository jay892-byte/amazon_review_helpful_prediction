import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data/raw/All_Beauty.jsonl"
OUTPUT_FILE = BASE_DIR / "data/processed/clean_reviews.jsonl"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

total = 0
kept = 0
missing_text = 0
invalid_vote = 0
label_0 = 0
label_1 = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        total += 1
        review = json.loads(line)

        # 1. 清洗评论文本
        text = review.get("text", "")

        if not isinstance(text, str) or not text.strip():
            missing_text += 1
            continue

        text = text.strip()

        # 2. 获取商品 ID
        product_id = review.get("parent_asin")

        if not product_id:
            continue

        # 3. 获取评分
        rating = review.get("rating")

        # 4. 获取有用票数
        vote = review.get("helpful_vote")

        if not isinstance(vote, int) or vote < 0:
            invalid_vote += 1
            continue

        # 5. 构造二分类标签
        label = 1 if vote >= 1 else 0

        if label == 1:
            label_1 += 1
        else:
            label_0 += 1

        # 6. 构造清洗后的记录
        clean_review = {
            "review_id": f"review_{total}",
            "parent_asin": product_id,
            "text": text,
            "title": review.get("title", ""),
            "rating": rating,
            "helpful_vote": vote,
            "label": label,
            "timestamp": review.get("timestamp"),
            "images": review.get("images", [])
        }

        # 7. 写入 JSONL 文件
        fout.write(
            json.dumps(clean_review, ensure_ascii=False) + "\n"
        )

        kept += 1

print("原始评论数：", total)
print("保留评论数：", kept)
print("删除空文本：", missing_text)
print("删除无效票数：", invalid_vote)
print("标签 0：", label_0)
print("标签 1：", label_1)