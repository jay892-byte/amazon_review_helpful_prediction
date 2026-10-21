
import json
from pathlib import Path
from collections import Counter

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data/raw/All_Beauty.jsonl"

total = 0
missing_text = 0
missing_product = 0
missing_rating = 0

ratings = Counter()
helpful_votes = Counter()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        review = json.loads(line)
        total += 1

        # 评论文本
        text = review.get("text")
        if not isinstance(text, str) or not text.strip():
            missing_text += 1

        # 商品 ID
        if not review.get("parent_asin"):
            missing_product += 1

        # 评分
        rating = review.get("rating")
        if rating is None:
            missing_rating += 1
        else:
            ratings[rating] += 1

        # 有用票数
        vote = review.get("helpful_vote")
        if isinstance(vote, int) and vote >= 0:
            helpful_votes[vote] += 1

print("评论总数：", total)
print("缺少评论文本：", missing_text)
print("缺少商品ID：", missing_product)
print("缺少评分：", missing_rating)

print("\n评分分布：")
for rating, count in sorted(ratings.items()):
    print(f"{rating} 星：{count}")

print("\n有用票数分布（前10种票数）：")
for vote, count in helpful_votes.most_common(10):
    print(f"{vote} 票：{count}")
