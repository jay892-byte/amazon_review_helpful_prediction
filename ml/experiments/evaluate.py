
import json
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from torch_geometric.data import HeteroData

from model import MultimodalGraphSAGE


# ========== 1. 路径配置 ==========

BASE_DIR = Path(__file__).resolve().parent.parent

GRAPH_FILE = BASE_DIR / "data/graph/review_product_graph.pt"
MAPPING_FILE = BASE_DIR / "data/graph/node_mapping.json"
REVIEW_FILE = BASE_DIR / "data/data_processed/clean_reviews.jsonl"

CHECKPOINT_FILE = BASE_DIR / "checkpoints/best_model.pth"
SPLIT_FILE = BASE_DIR / "checkpoints/experiment_split.json"

OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("使用设备：", DEVICE)


# ========== 2. 检查文件 ==========

for path in [
    GRAPH_FILE,
    MAPPING_FILE,
    REVIEW_FILE,
    CHECKPOINT_FILE,
    SPLIT_FILE
]:
    if not path.exists():
        raise FileNotFoundError(
            f"缺少必要文件：{path}"
        )


# ========== 3. 加载模型和图 ==========

# 只加载自己生成的可信 PyG 图文件
graph = torch.load(
    GRAPH_FILE,
    map_location="cpu",
    weights_only=False
)

graph.validate(raise_on_error=True)

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location="cpu",
    weights_only=True
)

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping = json.load(f)

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split = json.load(f)

review_ids = mapping["review_ids"]
product_ids = mapping["product_ids"]
parent_asins = mapping["review_parent_asins"]

assert len(review_ids) == graph["review"].num_nodes
assert len(product_ids) == graph["product"].num_nodes
assert len(parent_asins) == len(review_ids)

if len(set(review_ids)) != len(review_ids):
    raise ValueError("存在重复评论ID")

review_id_to_idx = {
    rid: i for i, rid in enumerate(review_ids)
}

product_id_to_idx = {
    pid: i for i, pid in enumerate(product_ids)
}


# ========== 4. 检查数据划分 ==========

train_ids = set(split["train_ids"])
val_ids = set(split["val_ids"])
test_ids = split["test_ids"]

if train_ids & val_ids:
    raise ValueError("训练集和验证集重叠")

if train_ids & set(test_ids):
    raise ValueError("训练集和测试集重叠")

if val_ids & set(test_ids):
    raise ValueError("验证集和测试集重叠")

if len(test_ids) != len(set(test_ids)):
    raise ValueError("测试集有重复评论ID")

if not test_ids:
    raise ValueError("测试集为空")

missing = set(test_ids) - set(review_id_to_idx)

if missing:
    raise ValueError(
        f"有 {len(missing)} 条测试评论不在图中"
    )

train_products = {
    parent_asins[review_id_to_idx[rid]]
    for rid in train_ids
}

val_products = {
    parent_asins[review_id_to_idx[rid]]
    for rid in val_ids
}

test_products = {
    parent_asins[review_id_to_idx[rid]]
    for rid in test_ids
}

assert train_products.isdisjoint(val_products)
assert train_products.isdisjoint(test_products)
assert val_products.isdisjoint(test_products)

print("商品分组无重叠：通过")


# ========== 5. 恢复数值标准化 ==========

mean = np.asarray(
    checkpoint["numeric_mean"],
    dtype=np.float32
)

scale = np.asarray(
    checkpoint["numeric_scale"],
    dtype=np.float32
)

if mean.shape != (4,) or scale.shape != (4,):
    raise ValueError("标准化参数维度错误")

if np.any(scale <= 0):
    raise ValueError("标准差参数无效")

needed_ids = set(test_ids)
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
    raise ValueError("部分测试评论缺少原始数据")


# ========== 6. 构建测试子图 ==========

test_indices = [
    review_id_to_idx[rid]
    for rid in test_ids
]

selected_products = sorted({
    parent_asins[i] for i in test_indices
})

product_indices = [
    product_id_to_idx[pid]
    for pid in selected_products
]

local_product_map = {
    pid: i for i, pid in enumerate(selected_products)
}

ridx = torch.tensor(test_indices, dtype=torch.long)
pidx = torch.tensor(product_indices, dtype=torch.long)

test_graph = HeteroData()

# 评论文本
test_graph["review"].text = graph["review"].text[ridx]

# 数值特征：使用训练时保存的均值和标准差
X = np.asarray(
    [raw_numeric[rid] for rid in test_ids],
    dtype=np.float32
)

X_scaled = (X - mean) / scale

test_graph["review"].numeric = torch.from_numpy(
    X_scaled.astype(np.float32)
)

test_graph["review"].y = graph["review"].y[ridx]
test_graph["review"].num_nodes = len(test_ids)

# 商品模态
test_graph["product"].text = graph["product"].text[pidx]
test_graph["product"].image = graph["product"].image[pidx]

test_graph["product"].image_mask = (
    graph["product"].image_mask[pidx]
)

test_graph["product"].num_nodes = len(selected_products)

# 重新生成测试子图的局部边索引
review_local = torch.arange(
    len(test_ids),
    dtype=torch.long
)

product_local = torch.tensor(
    [
        local_product_map[parent_asins[i]]
        for i in test_indices
    ],
    dtype=torch.long
)

edge_index = torch.stack(
    [review_local, product_local],
    dim=0
)

test_graph[
    "review", "belongs_to", "product"
].edge_index = edge_index

test_graph[
    "product", "has_review", "review"
].edge_index = edge_index.flip(0).contiguous()

test_graph.validate(raise_on_error=True)
test_graph = test_graph.to(DEVICE)

print("测试评论数：", len(test_ids))
print("测试商品数：", len(selected_products))


# ========== 7. 恢复模型 ==========

model = MultimodalGraphSAGE(
    hidden_dim=checkpoint["hidden_dim"],
    dropout=checkpoint["dropout"]
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ========== 8. 执行预测 ==========

with torch.inference_mode():

    logits = model(test_graph)

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= 0.5
    ).long()

y_true = test_graph["review"].y.cpu().numpy()
y_prob = probabilities.cpu().numpy()
y_pred = predictions.cpu().numpy()


# ========== 9. 计算评估指标 ==========

metrics = {
    "accuracy": float(
        accuracy_score(y_true, y_pred)
    ),
    "precision": float(
        precision_score(
            y_true, y_pred, zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_true, y_pred, zero_division=0
        )
    ),
    "f1": float(
        f1_score(
            y_true, y_pred, zero_division=0
        )
    )
}

# ROC-AUC 需要同时包含正负样本
if len(np.unique(y_true)) == 2:
    metrics["roc_auc"] = float(
        roc_auc_score(y_true, y_prob)
    )

    metrics["pr_auc"] = float(
        average_precision_score(y_true, y_prob)
    )
else:
    metrics["roc_auc"] = None # type: ignore
    metrics["pr_auc"] = None # type: ignore

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

print("\n===== 测试集评估结果 =====")

for key, value in metrics.items():
    if value is None:
        print(f"{key}: 不可计算")
    else:
        print(f"{key}: {value:.4f}")

print("\n混淆矩阵：")
print(cm)

print("\n分类报告：")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["无有用票", "有有用票"],
        zero_division=0
    )
)


# ========== 10. 保存评估结果 ==========

report = {
    "test_count": len(test_ids),
    "positive_count": int((y_true == 1).sum()),
    "negative_count": int((y_true == 0).sum()),
    "threshold": 0.5,
    "metrics": metrics,
    "confusion_matrix": cm.tolist()
}

with open(
    OUTPUT_DIR / "evaluation_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


# 保存逐条评论的预测结果
with open(
    OUTPUT_DIR / "test_predictions.csv",
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "review_id",
        "true_label",
        "predicted_label",
        "helpful_probability"
    ])

    for rid, true, pred, prob in zip(
        test_ids, y_true, y_pred, y_prob
    ):
        writer.writerow([
            rid,
            int(true),
            int(pred),
            float(prob)
        ])

print("\n评估完成")
print("报告：", OUTPUT_DIR / "evaluation_report.json")
print("预测：", OUTPUT_DIR / "test_predictions.csv")
