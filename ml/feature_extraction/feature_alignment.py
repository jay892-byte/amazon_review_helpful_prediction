
import json
from pathlib import Path

import torch
from tqdm import tqdm

# ========== 1. 路径配置 ==========

BASE_DIR = Path(__file__).resolve().parent.parent

REVIEW_FILE = (
    BASE_DIR / "data/data_processed/clean_reviews.jsonl"
)

FEATURE_DIR = BASE_DIR / "data/features"

REVIEW_TEXT_DIR = FEATURE_DIR / "review_text"
PRODUCT_TEXT_DIR = FEATURE_DIR / "product_text"
IMAGE_DIR = FEATURE_DIR / "image"

NUMERIC_FILE = (
    FEATURE_DIR / "numeric/numeric_features.pt"
)

OUTPUT_DIR = FEATURE_DIR / "aligned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 2. 参数配置 ==========

MAX_REVIEWS = 1000

TEXT_DIM = 384
IMAGE_DIM = 512
NUMERIC_DIM = 4

DEVICE = "cpu"


# ========== 3. 通用特征读取函数 ==========

def load_feature_data(path, id_keys):

    data = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

    if not isinstance(data, dict):
        raise ValueError(f"特征文件不是字典：{path}")

    ids = None

    for key in id_keys:
        if key in data:
            ids = data[key]
            break

    if ids is None:
        raise KeyError(
            f"{path} 中找不到 ID 字段，"
            f"实际字段：{list(data.keys())}"
        )

    features = None

    for key in ["features", "embeddings"]:
        if key in data:
            features = data[key]
            break

    if features is None:
        raise KeyError(
            f"{path} 中找不到特征字段，"
            f"实际字段：{list(data.keys())}"
        )

    if not isinstance(features, torch.Tensor):
        raise TypeError(f"特征不是 Tensor：{path}")

    if features.ndim != 2:
        raise ValueError(f"特征必须为二维：{path}")

    if len(ids) != features.shape[0]:
        raise ValueError(
            f"ID 数量和特征数量不一致：{path}"
        )

    return ids, features.float()


# ========== 4. 读取 1000 条评论文本特征 ==========

review_text_map = {}

review_files = sorted(
    REVIEW_TEXT_DIR.glob("features_*.pt")
)

if not review_files:
    raise FileNotFoundError(
        "没有找到评论文本特征文件"
    )

for file in tqdm(review_files, desc="读取评论文本特征"):

    ids, features = load_feature_data(
        file,
        ["review_ids", "ids"]
    )

    if features.shape[1] != TEXT_DIM:
        raise ValueError("评论文本特征维度错误")

    for i, rid in enumerate(ids):

        if len(review_text_map) >= MAX_REVIEWS:
            break

        if rid in review_text_map:
            raise ValueError(f"重复评论 ID：{rid}")

        review_text_map[rid] = features[i].clone()

    if len(review_text_map) >= MAX_REVIEWS:
        break

selected_review_ids = set(review_text_map)

print("选中评论数量：", len(selected_review_ids))


# ========== 5. 获取对应评论信息 ==========

review_info = {}
needed_product_ids = set()

with open(REVIEW_FILE, "r", encoding="utf-8") as f:

    for line in tqdm(f, desc="匹配评论信息"):

        review = json.loads(line)

        rid = review.get("review_id")

        if rid not in selected_review_ids:
            continue

        if rid in review_info:
            raise ValueError(f"重复评论 ID：{rid}")

        pid = review["parent_asin"]

        review_info[rid] = {
            "parent_asin": pid,
            "label": int(review["label"])
        }

        needed_product_ids.add(pid)

        if len(review_info) == len(selected_review_ids):
            break

missing_reviews = selected_review_ids - set(review_info)

if missing_reviews:
    raise ValueError(
        f"有 {len(missing_reviews)} 条评论"
        "无法匹配原始清洗数据"
    )

print("关联商品数量：", len(needed_product_ids))


# ========== 6. 读取数值特征 ==========

numeric_ids, numeric_features = load_feature_data(
    NUMERIC_FILE,
    ["ids", "review_ids"]
)

if numeric_features.shape[1] != NUMERIC_DIM:
    raise ValueError("数值特征维度错误")

numeric_map = {}

for i, rid in enumerate(numeric_ids):
    if rid in selected_review_ids:
        if rid in numeric_map:
            raise ValueError(f"重复数值特征：{rid}")

        numeric_map[rid] = numeric_features[i].clone()

missing_numeric = selected_review_ids - set(numeric_map)

if missing_numeric:
    raise ValueError(
        f"有 {len(missing_numeric)} 条评论缺少数值特征"
    )

del numeric_ids, numeric_features

print("数值特征匹配完成")


# ========== 7. 读取商品文本特征 ==========

product_text_map = {}

product_files = sorted(
    PRODUCT_TEXT_DIR.glob("features_*.pt")
)

if not product_files:
    raise FileNotFoundError(
        "没有商品文本特征文件，"
        "请先运行 product_feature_extraction.py"
    )

for file in tqdm(product_files, desc="读取商品文本特征"):

    ids, features = load_feature_data(
        file,
        ["product_ids", "ids"]
    )

    if features.shape[1] != TEXT_DIM:
        raise ValueError("商品文本特征维度错误")

    for i, pid in enumerate(ids):

        if pid not in needed_product_ids:
            continue

        if pid in product_text_map:
            raise ValueError(f"重复商品文本特征：{pid}")

        product_text_map[pid] = features[i].clone()

missing_product_text = (
    needed_product_ids - set(product_text_map)
)

if missing_product_text:
    raise ValueError(
        f"有 {len(missing_product_text)} 个商品"
        "缺少文本特征，请检查商品元数据匹配情况"
    )

print("商品文本特征匹配完成")


# ========== 8. 读取商品图像特征 ==========

image_map = {}

image_files = sorted(
    IMAGE_DIR.glob("product_image_*.pt")
)

for file in tqdm(image_files, desc="读取商品图像特征"):

    ids, features = load_feature_data(
        file,
        ["ids", "product_ids"]
    )

    if features.shape[1] != IMAGE_DIM:
        raise ValueError("图像特征维度错误")

    for i, pid in enumerate(ids):

        if pid not in needed_product_ids:
            continue

        if pid in image_map:
            raise ValueError(f"重复商品图像特征：{pid}")

        image_map[pid] = features[i].clone()

print("已有图像特征商品数：", len(image_map))


# ========== 9. 生成对齐后的评论数据 ==========

review_ids = []
parent_asins = []

review_text_features = []
review_numeric_features = []
labels = []

for rid in review_text_map:

    info = review_info[rid]

    review_ids.append(rid)
    parent_asins.append(info["parent_asin"])

    review_text_features.append(
        review_text_map[rid]
    )

    review_numeric_features.append(
        numeric_map[rid]
    )

    labels.append(info["label"])

review_output = {
    "review_ids": review_ids,
    "parent_asins": parent_asins,
    "text": torch.stack(review_text_features),
    "numeric": torch.stack(review_numeric_features),
    "labels": torch.tensor(labels, dtype=torch.long)
}

torch.save(
    review_output,
    OUTPUT_DIR / "review_features.pt"
)


# ========== 10. 生成对齐后的商品数据 ==========

product_ids = []

product_text_features = []
product_image_features = []
image_masks = []

missing_image = 0

for pid in sorted(needed_product_ids):

    product_ids.append(pid)

    product_text_features.append(
        product_text_map[pid]
    )

    if pid in image_map:

        product_image_features.append(
            image_map[pid]
        )

        image_masks.append(1.0)

    else:

        product_image_features.append(
            torch.zeros(IMAGE_DIM)
        )

        image_masks.append(0.0)

        missing_image += 1

product_output = {
    "product_ids": product_ids,
    "text": torch.stack(product_text_features),
    "image": torch.stack(product_image_features),
    "image_mask": torch.tensor(
        image_masks,
        dtype=torch.float32
    )
}

torch.save(
    product_output,
    OUTPUT_DIR / "product_features.pt"
)


# ========== 11. 保存检查报告 ==========

report = {
    "review_count": len(review_ids),
    "product_count": len(product_ids),
    "review_text_dim": TEXT_DIM,
    "numeric_dim": NUMERIC_DIM,
    "product_text_dim": TEXT_DIM,
    "image_dim": IMAGE_DIM,
    "products_with_image": len(image_map),
    "products_without_image": missing_image
}

with open(
    OUTPUT_DIR / "alignment_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n===== 特征对齐完成 =====")

for key, value in report.items():
    print(f"{key}: {value}")

print("\n评论特征形状：")
print("文本：", review_output["text"].shape)
print("数值：", review_output["numeric"].shape)

print("\n商品特征形状：")
print("文本：", product_output["text"].shape)
print("图像：", product_output["image"].shape)

print("\n输出目录：", OUTPUT_DIR)
