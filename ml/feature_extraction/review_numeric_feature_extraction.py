
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ========== 1. 路径配置 ==========

BASE_DIR = Path(__file__).resolve().parent.parent

REVIEW_FILE = (
    BASE_DIR / "data/data_processed/clean_reviews.jsonl"
)

SPLIT_DIR = BASE_DIR / "data/splits"

OUTPUT_DIR = BASE_DIR / "data/features/numeric"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = OUTPUT_DIR / "numeric_features.pt"
SCALER_FILE = OUTPUT_DIR / "scaler.pkl"

FEATURE_NAMES = [
    "rating",
    "word_count",
    "char_count",
    "title_word_count"
]


# ========== 2. 读取训练集 ID ==========

def load_ids(filename):
    with open(SPLIT_DIR / filename, "r",
              encoding="utf-8") as f:
        return set(json.load(f))


train_ids = load_ids("train_ids.json")
val_ids = load_ids("val_ids.json")
test_ids = load_ids("test_ids.json")

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

all_split_ids = train_ids | val_ids | test_ids

print("训练集：", len(train_ids))
print("验证集：", len(val_ids))
print("测试集：", len(test_ids))


# ========== 3. 提取数值特征 ==========

def extract_numeric_features(review):

    text = review.get("text") or ""
    title = review.get("title") or ""
    rating = review.get("rating")

    if not isinstance(text, str):
        text = ""

    if not isinstance(title, str):
        title = ""

    if rating is None:
        raise ValueError("存在缺失评分")

    rating = float(rating)

    word_count = len(text.split())
    char_count = len(text)
    title_word_count = len(title.split())

    return [
        rating,
        word_count,
        char_count,
        title_word_count
    ]


# ========== 4. 读取清洗后的评论 ==========

review_ids = []
numeric_features = []

seen_ids = set()

MAX_SAMPLES = 1000

with open(REVIEW_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="提取数值特征"):

        if len(review_ids) >= MAX_SAMPLES:
            break

        review = json.loads(line)
        review_id = review["review_id"]

        if review_id not in all_split_ids:
            continue

        if review_id in seen_ids:
            raise ValueError(f"重复评论ID：{review_id}")

        seen_ids.add(review_id)

        features = extract_numeric_features(review)

        review_ids.append(review_id)
        numeric_features.append(features)


X = np.asarray(numeric_features, dtype=np.float32)

print("原始特征形状：", X.shape)


# ========== 5. 检查数据完整性 ==========

if MAX_SAMPLES is None:
    if len(review_ids) != len(all_split_ids):
        missing = all_split_ids - seen_ids
        raise ValueError(
            f"存在 {len(missing)} 条未匹配评论"
        )

if not np.isfinite(X).all():
    raise ValueError("存在 NaN 或无穷大特征")


# ========== 6. 找到训练集对应的行 ==========

train_mask = np.array(
    [rid in train_ids for rid in review_ids],
    dtype=bool
)

X_train = X[train_mask]
if len(X_train) == 0:
    raise ValueError("当前1000条评论中没有训练集样本")
print("训练集特征形状：", X_train.shape)


# ========== 7. 使用训练集拟合标准化器 ==========

scaler = StandardScaler()

scaler.fit(X_train)

# 对全部样本应用训练集标准化参数
X_scaled = scaler.transform(X).astype(np.float32)

# 保存标准化器
joblib.dump(scaler, SCALER_FILE)


# ========== 8. 转换为 PyTorch Tensor ==========

numeric_tensor = torch.from_numpy(X_scaled)

torch.save(
    {
        "ids": review_ids,
        "embeddings": numeric_tensor,
        "feature_names": FEATURE_NAMES
    },
    FEATURE_FILE
)


# ========== 9. 输出结果 ==========

print("\n数值特征提取完成")
print("评论总数：", len(review_ids))
print("数值特征维度：", numeric_tensor.shape[1])
print("张量形状：", numeric_tensor.shape)
print("特征文件：", FEATURE_FILE)
print("标准化器：", SCALER_FILE)
