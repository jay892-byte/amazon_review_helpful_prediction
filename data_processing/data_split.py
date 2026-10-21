import numpy as np
import json
from pathlib import Path
from collections import Counter

from sklearn.model_selection import GroupShuffleSplit

# 1. 文件路径
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data/processed/clean_reviews.jsonl"
OUTPUT_DIR = BASE_DIR / "data/splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. 读取评论 ID、标签和商品 ID
review_ids = []
labels = []
product_ids = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        review = json.loads(line)

        review_ids.append(review["review_id"])
        labels.append(review["label"])
        product_ids.append(review["parent_asin"])

print("评论总数：", len(review_ids))
print("商品总数：", len(set(product_ids)))

# 3. 按商品划分训练集 80%、临时集 20%
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, temp_idx = next(
    splitter.split(
        np.array(review_ids),
        np.array(labels),
        groups=np.array(product_ids)
    )
)

# 4. 将临时集再次划分为验证集和测试集
temp_ids = [review_ids[i] for i in temp_idx]
temp_labels = [labels[i] for i in temp_idx]
temp_groups = [product_ids[i] for i in temp_idx]

splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.5,
    random_state=42
)

val_local_idx, test_local_idx = next(
    splitter2.split(
        np.array(temp_ids),
        np.array(temp_labels),
        groups=np.array(temp_groups)
    )
)

val_idx = [temp_idx[i] for i in val_local_idx]
test_idx = [temp_idx[i] for i in test_local_idx]

# 5. 获取三个数据集的评论 ID
train_ids = [review_ids[i] for i in train_idx]
val_ids = [review_ids[i] for i in val_idx]
test_ids = [review_ids[i] for i in test_idx]

# 6. 保存数据集 ID
def save_ids(filename, ids):
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)


save_ids("train_ids.json", train_ids)
save_ids("val_ids.json", val_ids)
save_ids("test_ids.json", test_ids)

# 7. 输出标签分布
def show_statistics(name, indices):
    counter = Counter(labels[i] for i in indices)

    total = len(indices)
    positive = counter.get(1, 0)
    negative = counter.get(0, 0)

    print(f"\n{name}")
    print("样本数：", total)
    print("标签 0：", negative)
    print("标签 1：", positive)

    if total:
        print(f"正样本比例：{positive / total:.2%}")


show_statistics("训练集", train_idx)
show_statistics("验证集", val_idx)
show_statistics("测试集", test_idx)

# 8. 检查数据集是否重叠
train_products = {product_ids[i] for i in train_idx}
val_products = {product_ids[i] for i in val_idx}
test_products = {product_ids[i] for i in test_idx}

assert not train_products & val_products
assert not train_products & test_products
assert not val_products & test_products

assert len(train_ids) + len(val_ids) + len(test_ids) == len(review_ids)

assert len(set(train_ids)) == len(train_ids)
assert len(set(val_ids)) == len(val_ids)
assert len(set(test_ids)) == len(test_ids)

print("\n数据划分完成")
print("训练集：", len(train_ids))
print("验证集：", len(val_ids))
print("测试集：", len(test_ids))
print("商品分组无重叠：通过")