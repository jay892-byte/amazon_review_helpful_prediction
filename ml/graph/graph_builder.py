
import json
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

# ========== 1. 路径配置 ==========

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_DIR = BASE_DIR / "data/features/aligned"
OUTPUT_DIR = BASE_DIR / "data/graph"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REVIEW_FILE = FEATURE_DIR / "review_features.pt"
PRODUCT_FILE = FEATURE_DIR / "product_features.pt"

GRAPH_FILE = OUTPUT_DIR / "review_product_graph.pt"
REPORT_FILE = OUTPUT_DIR / "graph_report.json"


# ========== 2. 读取已对齐特征 ==========

reviews = torch.load(
    REVIEW_FILE,
    map_location="cpu",
    weights_only=True
)

products = torch.load(
    PRODUCT_FILE,
    map_location="cpu",
    weights_only=True
)

review_ids = reviews["review_ids"]
parent_asins = reviews["parent_asins"]
product_ids = products["product_ids"]

print("评论节点数：", len(review_ids))
print("商品节点数：", len(product_ids))


# ========== 3. 检查节点及特征 ==========

num_reviews = len(review_ids)
num_products = len(product_ids)

assert num_reviews > 0
assert num_products > 0

assert len(set(review_ids)) == num_reviews
assert len(set(product_ids)) == num_products

assert len(parent_asins) == num_reviews

assert reviews["text"].shape == (
    num_reviews, 384
)

assert reviews["numeric"].shape == (
    num_reviews, 4
)

assert reviews["labels"].shape == (
    num_reviews,
)

assert products["text"].shape == (
    num_products, 384
)

assert products["image"].shape == (
    num_products, 512
)

assert products["image_mask"].shape == (
    num_products,
)

assert torch.isin(
    reviews["labels"],
    torch.tensor([0, 1])
).all()

for name, tensor in [
    ("review_text", reviews["text"]),
    ("review_numeric", reviews["numeric"]),
    ("product_text", products["text"]),
    ("product_image", products["image"])
]:
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"{name} 包含 NaN 或 Inf"
        )


# ========== 4. 建立节点索引 ==========

review_id_to_idx = {
    rid: idx
    for idx, rid in enumerate(review_ids)
}

product_id_to_idx = {
    pid: idx
    for idx, pid in enumerate(product_ids)
}

# 为评论节点建立对应商品的边
review_indices = []
product_indices = []

for review_idx, product_id in enumerate(parent_asins):

    if product_id not in product_id_to_idx:
        raise ValueError(
            f"找不到商品节点：{product_id}"
        )

    product_idx = product_id_to_idx[product_id]

    review_indices.append(review_idx)
    product_indices.append(product_idx)


# ========== 5. 初始化异构图 ==========

data = HeteroData()

# ========== 6. 评论节点 ==========

data["review"].num_nodes = num_reviews

data["review"].text = reviews["text"].float()
data["review"].numeric = reviews["numeric"].float()

# 标签单独保存，不参与特征融合
data["review"].y = reviews["labels"].long()


# ========== 7. 商品节点 ==========

data["product"].num_nodes = num_products

data["product"].text = products["text"].float()
data["product"].image = products["image"].float()

data["product"].image_mask = (
    products["image_mask"].float()
)


# ========== 8. 构建评论 → 商品的边 ==========

edge_index = torch.tensor(
    [review_indices, product_indices],
    dtype=torch.long
)

data[
    "review", "belongs_to", "product"
].edge_index = edge_index


# ========== 9. 构建商品 → 评论的反向边 ==========

reverse_edge_index = edge_index.flip(0).contiguous()

data[
    "product", "has_review", "review"
].edge_index = reverse_edge_index


# ========== 10. 图完整性检查 ==========

data.validate(raise_on_error=True)

assert edge_index.shape == (2, num_reviews)

assert torch.equal(
    reverse_edge_index,
    edge_index.flip(0)
)

print("\n图节点类型：", data.node_types)
print("图边类型：", data.edge_types)

print("评论 → 商品边数：", edge_index.shape[1])
print("商品 → 评论边数：", reverse_edge_index.shape[1])


# ========== 11. 保存图数据 ==========

torch.save(data, GRAPH_FILE)

# 分开保存索引，方便后续定位节点
with open(
    OUTPUT_DIR / "node_mapping.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "review_ids": review_ids,
            "product_ids": product_ids,
            "review_parent_asins": parent_asins
        },
        f,
        ensure_ascii=False
    )


# ========== 12. 保存图统计报告 ==========

labels = data["review"].y

report = {
    "review_nodes": num_reviews,
    "product_nodes": num_products,
    "belongs_to_edges": edge_index.shape[1],
    "has_review_edges": reverse_edge_index.shape[1],
    "label_0": int((labels == 0).sum()),
    "label_1": int((labels == 1).sum()),
    "products_with_image": int(
        data["product"].image_mask.sum()
    ),
    "products_without_image": int(
        (data["product"].image_mask == 0).sum()
    )
}

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(report, f, indent=2, ensure_ascii=False)


print("\n===== 图构建完成 =====")

for key, value in report.items():
    print(f"{key}: {value}")

print("\n保存图文件：", GRAPH_FILE)
