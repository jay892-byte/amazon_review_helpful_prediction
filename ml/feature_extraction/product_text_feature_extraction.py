
import json
from pathlib import Path

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 1. 文件路径
BASE_DIR = Path(__file__).resolve().parent.parent

REVIEW_FILE = BASE_DIR / "data/data_processed/clean_reviews.jsonl"
PRODUCT_FILE = BASE_DIR / "data/data_processed/clean_products.jsonl"

REVIEW_FEATURE_DIR = BASE_DIR / "data/features/review_text"
OUTPUT_DIR = BASE_DIR / "data/features/product_text"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. 参数
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

print("使用设备：", DEVICE)

# 3. 获取已经提取特征的评论 ID
review_ids = set()

for file in sorted(REVIEW_FEATURE_DIR.glob("features_*.pt")):
    data = torch.load(
        file,
        map_location="cpu",
        weights_only=True
    )
    review_ids.update(data["review_ids"])

if not review_ids:
    raise ValueError("没有找到评论文本特征")

print("待处理评论数量：", len(review_ids))

# 4. 找出这批评论涉及的商品
product_ids = set()

with open(REVIEW_FILE, "r", encoding="utf-8") as f:
    for line in f:
        review = json.loads(line)

        if review["review_id"] in review_ids:
            product_ids.add(review["parent_asin"])

print("涉及商品数量：", len(product_ids))

# 5. 清洗商品文本
def to_text(value):
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        return " ".join(
            str(x).strip()
            for x in value
            if x is not None
        )

    return ""

# 6. 加载 MiniLM
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# 7. 提取匹配商品的文本
ids = []
texts = []
seen = set()

with open(PRODUCT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="读取商品"):
        product = json.loads(line)

        pid = product.get("parent_asin")

        if pid not in product_ids or pid in seen:
            continue

        seen.add(pid)

        title = to_text(product.get("title"))
        description = to_text(product.get("description"))
        features = to_text(product.get("features"))

        text = f"{title} {description} {features}".strip()

        ids.append(pid)
        texts.append(text if text else "unknown")

# 8. 批量提取向量
if not texts:
    raise ValueError("没有匹配到商品元数据")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
).cpu()

# 9. 保存商品特征
output_file = OUTPUT_DIR / "features_0000.pt"

torch.save(
    {
        "product_ids": ids,
        "features": embeddings
    },
    output_file
)

print("\n商品文本特征提取完成")
print("评论涉及商品数：", len(product_ids))
print("匹配商品数：", len(ids))
print("未匹配商品数：", len(product_ids) - len(ids))
print("特征形状：", tuple(embeddings.shape))
print("保存位置：", output_file)
