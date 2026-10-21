
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# ========== 1. 文件路径 ==========

BASE_DIR = Path(__file__).resolve().parent.parent

ALIGNED_DIR = BASE_DIR / "data/features/aligned"
REVIEW_FILE = BASE_DIR / "data/data_processed/clean_reviews.jsonl"

SPLIT_FILE = BASE_DIR / "checkpoints/experiment_split.json"
CHECKPOINT_FILE = BASE_DIR / "checkpoints/best_model.pth"

OUTPUT_DIR = BASE_DIR / "results/baselines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ========== 2. 加载数据 ==========

reviews = torch.load(
    ALIGNED_DIR / "review_features.pt",
    map_location="cpu",
    weights_only=True
)

products = torch.load(
    ALIGNED_DIR / "product_features.pt",
    map_location="cpu",
    weights_only=True
)

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    splits = json.load(f)

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location="cpu",
    weights_only=True
)

review_ids = reviews["review_ids"]
product_ids = products["product_ids"]

review_to_idx = {
    rid: i for i, rid in enumerate(review_ids)
}

product_to_idx = {
    pid: i for i, pid in enumerate(product_ids)
}

if len(review_to_idx) != len(review_ids):
    raise ValueError("评论ID重复")

if len(product_to_idx) != len(product_ids):
    raise ValueError("商品ID重复")


# ========== 3. 恢复数值标准化参数 ==========

mean = np.asarray(
    checkpoint["numeric_mean"], dtype=np.float32
)

scale = np.asarray(
    checkpoint["numeric_scale"], dtype=np.float32
)

if mean.shape != (4,) or scale.shape != (4,):
    raise ValueError("标准化参数不正确")

# 从原始评论重新生成数值特征
needed_ids = set(review_ids)
raw_numeric = {}

with open(REVIEW_FILE, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        rid = r["review_id"]

        if rid not in needed_ids:
            continue

        text = r.get("text") or ""
        title = r.get("title") or ""

        raw_numeric[rid] = [
            float(r["rating"]),
            len(text.split()),
            len(text),
            len(title.split())
        ]

        if len(raw_numeric) == len(needed_ids):
            break

if len(raw_numeric) != len(needed_ids):
    raise ValueError("缺少评论原始数值特征")


# ========== 4. 对齐所有输入特征 ==========

review_text = reviews["text"].numpy()
product_text = products["text"].numpy()
product_image = products["image"].numpy()
image_mask = products["image_mask"].numpy()

numeric_raw = np.asarray(
    [raw_numeric[rid] for rid in review_ids],
    dtype=np.float32
)

numeric = (numeric_raw - mean) / scale

# 商品顺序按照评论对应的 parent_asin 排列
product_indices = np.asarray([
    product_to_idx[pid]
    for pid in reviews["parent_asins"]
])

X = np.concatenate(
    [
        review_text,
        numeric,
        product_text[product_indices],
        product_image[product_indices],
        image_mask[product_indices].reshape(-1, 1)
    ],
    axis=1
).astype(np.float32)

y = reviews["labels"].numpy().astype(int)

if not np.isfinite(X).all():
    raise ValueError("特征包含 NaN 或 Inf")

assert X.shape == (len(review_ids), 1285)

print("特征形状：", X.shape)
print("标签形状：", y.shape)


# ========== 5. 复用 GraphSAGE 的数据划分 ==========

def get_indices(ids):
    missing = set(ids) - set(review_to_idx)

    if missing:
        raise ValueError(
            f"存在 {len(missing)} 条未找到的评论"
        )

    return np.asarray(
        [review_to_idx[rid] for rid in ids],
        dtype=np.int64
    )


train_idx = get_indices(splits["train_ids"])
val_idx = get_indices(splits["val_ids"])
test_idx = get_indices(splits["test_ids"])

if len(set(train_idx) & set(val_idx)):
    raise ValueError("训练集与验证集重叠")

if len(set(train_idx) & set(test_idx)):
    raise ValueError("训练集与测试集重叠")

if len(set(val_idx) & set(test_idx)):
    raise ValueError("验证集与测试集重叠")

if len(set(train_idx)) != len(train_idx):
    raise ValueError("训练集存在重复ID")

if len(set(val_idx)) != len(val_idx):
    raise ValueError("验证集存在重复ID")

if len(set(test_idx)) != len(test_idx):
    raise ValueError("测试集存在重复ID")

all_products = reviews["parent_asins"]

group_sets = [
    {all_products[i] for i in indices}
    for indices in (train_idx, val_idx, test_idx)
]

assert group_sets[0].isdisjoint(group_sets[1])
assert group_sets[0].isdisjoint(group_sets[2])
assert group_sets[1].isdisjoint(group_sets[2])

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

if len(np.unique(y_train)) != 2:
    raise ValueError("训练集必须包含两种标签")

print("训练样本数：", len(train_idx))
print("验证样本数：", len(val_idx))
print("测试样本数：", len(test_idx))


# ========== 6. 定义基线模型 ==========

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    ),

    "MLP": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=200,
        early_stopping=False,
        random_state=SEED
    )
}


# ========== 7. 计算评估指标 ==========

def evaluate(y_true, probabilities):
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    result = {
        "Accuracy": accuracy_score(
            y_true, predictions
        ),
        "Precision": precision_score(
            y_true, predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions,
            zero_division=0
        )
    }

    if len(np.unique(y_true)) == 2:
        result["ROC-AUC"] = roc_auc_score(
            y_true, probabilities
        )

        result["PR-AUC"] = average_precision_score(
            y_true, probabilities
        )
    else:
        result["ROC-AUC"] = np.nan
        result["PR-AUC"] = np.nan

    return result


# ========== 8. 训练并比较模型 ==========

results = []

for name, model in models.items():

    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    val_metrics = evaluate(y_val, val_prob)
    test_metrics = evaluate(y_test, test_prob)

    print("验证集 F1：", round(val_metrics["F1"], 4))
    print("测试集 F1：", round(test_metrics["F1"], 4))

    results.append({
        "Model": name,
        "Val_F1": val_metrics["F1"],
        **test_metrics
    })


# ========== 9. 保存结果 ==========

df = pd.DataFrame(results)

output_file = OUTPUT_DIR / "baseline_results.csv"
df.to_csv(output_file, index=False)

print("\n===== 基线模型对比 =====")
print(df.to_string(index=False))

print("\n结果已保存：", output_file)

