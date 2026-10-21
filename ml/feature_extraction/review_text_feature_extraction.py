
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 1. 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data/data_processed/clean_reviews.jsonl"
OUTPUT_DIR = BASE_DIR / "data/features/review_text"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. 模型及运行参数
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
SHARD_SIZE = 1000


MAX_REVIEWS = 1000

print("使用设备：", DEVICE)

# 3. 加载 MiniLM
model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

# 4. 编码并保存一个数据块
def save_shard(texts, review_ids, shard_index):
    if not texts:
        return

    # MiniLM 编码
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # 转移至 CPU 保存
    embeddings = embeddings.cpu()

    output_file = OUTPUT_DIR / (
        f"features_{shard_index:04d}.pt"
    )

    torch.save(
        {
            "review_ids": review_ids,
            "features": embeddings
        },
        output_file
    )

    print(
        f"保存 {output_file.name}，"
        f"形状：{tuple(embeddings.shape)}"
    )


# 5. 逐行读取评论并批量提取特征
texts = []
review_ids = []

total = 0
shard_index = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in tqdm(f, desc="读取评论"):

        if MAX_REVIEWS is not None:
            if total >= MAX_REVIEWS:
                break

        review = json.loads(line)

        text = review["text"]
        review_id = review["review_id"]

        texts.append(text)
        review_ids.append(review_id)

        total += 1

        # 累计达到指定数量后保存
        if len(texts) >= SHARD_SIZE:

            save_shard(
                texts,
                review_ids,
                shard_index
            )

            texts = []
            review_ids = []
            shard_index += 1

# 6. 保存最后不足一块的评论
if texts:
    save_shard(
        texts,
        review_ids,
        shard_index
    )

print("\n评论文本特征提取完成")
print("处理评论数：", total)
print("特征维度：", model.get_sentence_embedding_dimension())
