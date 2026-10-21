
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from torch_geometric.data import HeteroData

from model import MultimodalGraphSAGE


# ========== 1. 配置 ==========

BASE_DIR = Path(__file__).resolve().parent.parent

GRAPH_FILE = BASE_DIR / "data/graph/review_product_graph.pt"
MAPPING_FILE = BASE_DIR / "data/graph/node_mapping.json"
REVIEW_FILE = BASE_DIR / "data/data_processed/clean_reviews.jsonl"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 15
HIDDEN_DIM = 128
DROPOUT = 0.3

print("使用设备：", DEVICE)


# ========== 2. 设置随机种子 ==========

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ========== 3. 加载图与节点映射 ==========

# 只加载自己生成的可信 PyG 图文件
data = torch.load(
    GRAPH_FILE,
    map_location="cpu",
    weights_only=False
)

data.validate(raise_on_error=True)

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping = json.load(f)

review_ids = mapping["review_ids"]
product_ids = mapping["product_ids"]
parent_asins = mapping["review_parent_asins"]

labels = data["review"].y.cpu().numpy()

assert len(review_ids) == data["review"].num_nodes
assert len(product_ids) == data["product"].num_nodes
assert len(parent_asins) == len(review_ids)

# ========== 4. 按商品分组划分 ==========

indices = np.arange(len(review_ids))
groups = np.asarray(parent_asins)

if len(set(groups)) < 3:
    raise ValueError("商品数量过少，无法划分三个数据集")

split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=SEED
)

train_idx, temp_idx = next(
    split1.split(indices, labels, groups)
)

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.5,
    random_state=SEED
)

val_local, test_local = next(
    split2.split(
        temp_idx,
        labels[temp_idx],
        groups[temp_idx]
    )
)

val_idx = temp_idx[val_local]
test_idx = temp_idx[test_local]

for name, idx in [
    ("训练集", train_idx),
    ("验证集", val_idx),
    ("测试集", test_idx)
]:
    print(
        name,
        "评论数：", len(idx),
        "正样本数：", int(labels[idx].sum())
    )

assert set(groups[train_idx]).isdisjoint(groups[val_idx])
assert set(groups[train_idx]).isdisjoint(groups[test_idx])
assert set(groups[val_idx]).isdisjoint(groups[test_idx])

if len(set(labels[train_idx])) < 2:
    raise ValueError("训练集只有一种标签，请调整划分")


# ========== 5. 重建原始数值特征 ==========

needed = set(review_ids)
raw_numeric = {}

with open(REVIEW_FILE, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        rid = r["review_id"]

        if rid not in needed:
            continue

        text = r.get("text") or ""
        title = r.get("title") or ""

        raw_numeric[rid] = [
            float(r["rating"]),
            len(text.split()),
            len(text),
            len(title.split())
        ]

        if len(raw_numeric) == len(needed):
            break

if len(raw_numeric) != len(needed):
    raise ValueError("部分评论缺少原始数值特征")

X = np.asarray(
    [raw_numeric[rid] for rid in review_ids],
    dtype=np.float32
)

# 只使用当前训练集拟合
scaler = StandardScaler()
scaler.fit(X[train_idx])

X = scaler.transform(X).astype(np.float32)

# 修改内存中的图，不覆盖原图文件
data["review"].numeric = torch.from_numpy(X)


# ========== 6. 构建独立子图 ==========

def build_subgraph(review_indices):

    review_indices = np.asarray(
        review_indices, dtype=np.int64
    )

    # 当前子图涉及的商品
    selected_products = sorted({
        parent_asins[i] for i in review_indices
    })

    product_global_map = {
        pid: i for i, pid in enumerate(product_ids)
    }

    product_local_map = {
        pid: i for i, pid in enumerate(selected_products)
    }

    product_indices = torch.tensor(
        [product_global_map[pid] for pid in selected_products],
        dtype=torch.long
    )

    ridx = torch.as_tensor(review_indices, dtype=torch.long)

    sub = HeteroData()

    # 评论节点
    sub["review"].text = data["review"].text[ridx]
    sub["review"].numeric = data["review"].numeric[ridx]
    sub["review"].y = data["review"].y[ridx]
    sub["review"].num_nodes = len(review_indices)

    # 商品节点
    sub["product"].text = data["product"].text[product_indices]
    sub["product"].image = data["product"].image[product_indices]
    sub["product"].image_mask = (
        data["product"].image_mask[product_indices]
    )
    sub["product"].num_nodes = len(selected_products)

    # 局部边索引
    review_local = torch.arange(
        len(review_indices), dtype=torch.long
    )

    product_local = torch.tensor(
        [
            product_local_map[parent_asins[i]]
            for i in review_indices
        ],
        dtype=torch.long
    )

    edge_index = torch.stack(
        [review_local, product_local]
    )

    sub[
        "review", "belongs_to", "product"
    ].edge_index = edge_index

    sub[
        "product", "has_review", "review"
    ].edge_index = edge_index.flip(0).contiguous()

    sub.validate(raise_on_error=True)

    return sub.to(DEVICE)


train_graph = build_subgraph(train_idx)
val_graph = build_subgraph(val_idx)
test_graph = build_subgraph(test_idx)

print("\n三个独立子图构建成功")


# ========== 7. 初始化模型 ==========

model = MultimodalGraphSAGE(
    hidden_dim=HIDDEN_DIM,
    dropout=DROPOUT
).to(DEVICE)

# 使用训练集正负样本数量计算权重
y_train = train_graph["review"].y

positive = int((y_train == 1).sum())
negative = int((y_train == 0).sum())

pos_weight = torch.tensor(
    [negative / positive],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ========== 8. 评估函数 ==========

@torch.no_grad()
def evaluate(graph):
    model.eval()

    logits = model(graph)
    y = graph["review"].y.float()

    loss = criterion(logits, y)

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    yt = y.cpu().numpy().astype(int)
    yp = preds.cpu().numpy()

    return {
        "loss": float(loss.item()),
        "accuracy": accuracy_score(yt, yp),
        "precision": precision_score(
            yt, yp, zero_division=0
        ),
        "recall": recall_score(
            yt, yp, zero_division=0
        ),
        "f1": f1_score(
            yt, yp, zero_division=0
        )
    }


# ========== 9. 训练循环 ==========

best_f1 = -1.0
best_loss = float("inf")
wait = 0

CHECKPOINT_FILE = CHECKPOINT_DIR / "best_model.pth"

for epoch in range(1, EPOCHS + 1):

    model.train()
    optimizer.zero_grad()

    logits = model(train_graph)
    y = train_graph["review"].y.float()

    loss = criterion(logits, y)

    loss.backward()
    optimizer.step()

    val_metrics = evaluate(val_graph)

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {loss.item():.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f}"
    )

    f1 = val_metrics["f1"]
    val_loss = val_metrics["loss"]

    improved = (
        f1 > best_f1 + 1e-6
        or (
            abs(f1 - best_f1) <= 1e-6
            and val_loss < best_loss
        )
    )

    if improved:
        best_f1 = f1
        best_loss = val_loss
        wait = 0

        assert scaler.mean_ is not None
        assert scaler.scale_ is not None
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_f1": best_f1,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT,
                "numeric_mean": scaler.mean_.tolist(),
                "numeric_scale": scaler.scale_.tolist(),
                "seed": SEED
            },
            CHECKPOINT_FILE
        )

        print("  → 保存最佳模型")

    else:
        wait += 1

    if wait >= PATIENCE:
        print("\nEarly Stopping")
        break


# ========== 10. 保存划分 ID ==========

split_ids = {
    "train_ids": [review_ids[i] for i in train_idx],
    "val_ids": [review_ids[i] for i in val_idx],
    "test_ids": [review_ids[i] for i in test_idx]
}

with open(
    CHECKPOINT_DIR / "experiment_split.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(split_ids, f, ensure_ascii=False, indent=2)

print("\n训练结束")
print("最佳验证集 F1：", round(best_f1, 4))
print("最佳模型：", CHECKPOINT_FILE)
print("测试集暂未评估，请在 evaluate.py 中进行最终测试")
